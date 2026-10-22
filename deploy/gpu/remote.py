"""Password SSH deployment; no benchmark runtime dependency or automatic GPU workload."""

import argparse
import base64
import getpass
import hashlib
from pathlib import Path, PurePosixPath
import shlex
import sys
import time
import uuid

import paramiko
import yaml

BASE = Path(__file__).resolve().parent
FILES = ("dcgm.sh", "dcgm-counters.csv", "nsight.sh", "nsight.Dockerfile", "capture.py", ".dockerignore")


def load_config(path):
    value = yaml.safe_load(path.read_text())
    required = {"ssh", "runtime", "remote_dir", "gpu_bind_ip", "dcgm_image",
                "inference_image", "nsight_image", "nsight_deb", "nerdctl_namespace"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Configuration must contain exactly the fields in remote.example.yaml")
    ssh = value["ssh"]
    if not isinstance(ssh, dict) or set(ssh) != {"host", "port", "username", "password"}:
        raise ValueError("ssh requires host, port, username, password")
    for key in ("host", "username"):
        if not isinstance(ssh[key], str) or not ssh[key].strip():
            raise ValueError(f"ssh.{key} must be a nonempty string")
    if type(ssh["port"]) is not int or not 1 <= ssh["port"] <= 65535:
        raise ValueError("ssh.port must be an integer in 1..65535")
    if not isinstance(ssh["password"], str):
        raise ValueError("ssh.password must be a quoted string (empty prompts securely)")
    if ssh["password"] and path.stat().st_mode & 0o077:
        raise ValueError("Password file must be private: chmod 600 <config>")
    for key in required - {"ssh"}:
        if not isinstance(value[key], str) or not value[key].strip() or "\x00" in value[key]:
            raise ValueError(f"{key} must be a nonempty string without NUL")
    if value["runtime"] not in {"docker", "nerdctl"}:
        raise ValueError("runtime must be docker or nerdctl")
    remote = PurePosixPath(value["remote_dir"])
    if not remote.is_absolute() or ".." in remote.parts or str(remote) == "/":
        raise ValueError("remote_dir must be an absolute, non-root server directory without ..")
    deb = Path(value["nsight_deb"]).expanduser()
    if not deb.is_absolute():
        deb = path.resolve().parent / deb
    if deb.suffix != ".deb" or not deb.is_file() or deb.stat().st_size == 0:
        raise ValueError("nsight_deb must reference an existing, nonempty local .deb")
    value["nsight_deb"] = deb
    return value


class ConfirmHostKey(paramiko.MissingHostKeyPolicy):
    def __init__(self, known_hosts):
        self.known_hosts = known_hosts

    def missing_host_key(self, client, hostname, key):
        digest = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
        if not sys.stdin.isatty():
            raise paramiko.SSHException("Unknown SSH host; run interactively to verify its fingerprint")
        answer = input(f"First connection to {hostname}: {key.get_name()} SHA256:{digest}\n"
                       "Verify with the server owner. Trust this key? [yes/no] ")
        if answer != "yes":
            raise paramiko.SSHException("SSH host key was not accepted")
        self.known_hosts.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.known_hosts.touch(mode=0o600, exist_ok=True)
        client.get_host_keys().add(hostname, key.get_name(), key)
        client.save_host_keys(str(self.known_hosts))


def run(client, argv, *, env=None, timeout=3600):
    command = shlex.join(["env", *[f"{k}={v}" for k, v in (env or {}).items()], *argv])
    with client.get_transport().open_session(timeout=30) as channel:
        channel.set_combine_stderr(True)
        channel.exec_command(command)
        channel.shutdown_write()
        deadline = time.monotonic() + timeout
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Remote operation timed out; inspect server state before retrying")
            if channel.recv_ready():
                sys.stdout.buffer.write(channel.recv(65536))
                sys.stdout.buffer.flush()
            elif channel.exit_status_ready():
                status = channel.recv_exit_status()
                if status != 0:
                    raise RuntimeError(f"Remote operation failed (exit {status})")
                return
            else:
                time.sleep(0.05)


def deploy(client, config):
    runtime = config["runtime"]
    engine = [runtime]
    if runtime == "nerdctl":
        engine += ["--namespace", config["nerdctl_namespace"]]
    print("Checking NVIDIA driver, container runtime and deployment prerequisites", flush=True)
    run(client, ["bash", "-c", "set -e; command -v bash curl nvidia-smi; nvidia-smi; "
                 + shlex.join([*engine, "info"]) + "; if "
                 + shlex.join([*engine, "inspect", "perf-dcgm"])
                 + " >/dev/null 2>&1; then echo 'perf-dcgm already exists; inspect it before deploying'; exit 1; fi"],
        timeout=120)
    folder = str(PurePosixPath(config["remote_dir"]) / ("deployment-" + uuid.uuid4().hex))
    run(client, ["mkdir", "-p", "--", folder], timeout=30)
    print(f"Uploading deployment files to {folder}", flush=True)
    with client.open_sftp() as sftp:
        sftp.get_channel().settimeout(120)
        for name in FILES:
            sftp.put(str(BASE / name), f"{folder}/{name}")
        sftp.put(str(config["nsight_deb"]), f"{folder}/nsight-systems-cli.deb")
    env = {"NERDCTL_NAMESPACE": config["nerdctl_namespace"],
           "GPU_BIND_IP": config["gpu_bind_ip"], "DCGM_EXPORTER_IMAGE": config["dcgm_image"],
           "INFERENCE_IMAGE": config["inference_image"], "NSIGHT_IMAGE": config["nsight_image"]}
    print("Building Nsight diagnostic image (no inference workload is launched)", flush=True)
    run(client, ["bash", f"{folder}/nsight.sh", runtime, "build"], env=env)
    print("Starting DCGM and waiting for actual GPU metric samples", flush=True)
    run(client, ["bash", f"{folder}/dcgm.sh", runtime, "up"], env=env, timeout=900)
    print(f"Deployment complete. DCGM: {config['gpu_bind_ip']}:9400\n"
          f"Nsight image: {config['nsight_image']}\nRemote scripts: {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="local private YAML; see remote.example.yaml")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        ssh = config["ssh"]
        password = ssh["password"] or getpass.getpass("GPU SSH password: ")
        known_hosts = Path.home() / ".ssh/known_hosts"
        with paramiko.SSHClient() as client:
            if known_hosts.exists():
                client.load_host_keys(str(known_hosts))
            client.set_missing_host_key_policy(ConfirmHostKey(known_hosts))
            client.connect(hostname=ssh["host"], port=ssh["port"], username=ssh["username"],
                           password=password, allow_agent=False, look_for_keys=False,
                           timeout=30, banner_timeout=30, auth_timeout=30)
            transport = client.get_transport()
            if transport is None:
                raise RuntimeError("SSH transport unavailable after connecting")
            transport.set_keepalive(30)
            deploy(client, config)
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"GPU deployment failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, paramiko.SSHException, yaml.YAMLError) as exc:
        # Do not echo malformed YAML, credentials or remote command arguments.
        print(f"GPU deployment failed ({type(exc).__name__}). Check config, SSH authentication/host key, "
              "and preceding remote output. Uploaded files/containers are retained for inspection.",
              file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("Deployment interrupted; inspect remote state before retrying.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
