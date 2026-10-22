"""Pipeline hosts and contracts.

Import concrete hosts and contracts from their defining modules.  Keeping this
package initializer empty prevents the public contract modules from importing
the Coordinator and, through it, every Stage implementation.
"""

__all__: list[str] = []
