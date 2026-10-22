"""Keep fixed Grafana dashboard navigation inside the same-origin embedded app."""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def update(directory, base):
    parts = urlsplit(base)
    if (parts.scheme not in {'http', 'https'} or not parts.hostname or parts.username is not None
            or parts.password is not None or parts.path not in {'', '/'} or parts.query or parts.fragment
            or any(c.isspace() for c in base)):
        raise ValueError('WORKBENCH_PUBLIC_URL must be an HTTP(S) origin without credentials, path, query or fragment')
    parts.port  # Reject malformed or out-of-range ports before changing dashboards.
    def navigation(value):
        if isinstance(value, dict):
            url = value.get('url')
            if isinstance(url, str) and 'targetBlank' in value:
                value['targetBlank'] = False
            for child in value.values():
                navigation(child)
        elif isinstance(value, list):
            for child in value:
                navigation(child)

    for name in ('runs.json', 'overview.json'):
        path = Path(directory) / name
        value = json.loads(path.read_text())
        for link in value.get('links', []):
            if link['url'].startswith(('/d/', '/explore?')):
                link['url'] = '/grafana' + link['url']
        navigation(value)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    update('/etc/grafana/dashboards', os.environ['WORKBENCH_PUBLIC_URL'])
