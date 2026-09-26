"""
Pure-Python and Accelerated Ed25519 (RFC 8032) Cryptographic Provider.

Implements high-speed, non-repudiable asymmetric digital signatures over Edwards-curve 25519.
Operates fully offline with zero required external dependencies (pure Python fallback using
hashlib.sha512 and extended projective coordinates), while automatically utilizing OpenSSL
C-bindings if the `cryptography` library is installed in the runtime environment.
"""

from __future__ import annotations

import hashlib
import secrets

# Optional hardware/C acceleration via cryptography library
try:
    from cryptography.hazmat.primitives.asymmetric import ed25519 as _crypto_ed25519
    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _crypto_ed25519 = None
    _HAS_CRYPTOGRAPHY = False

# RFC 8032 Ed25519 Curve Parameters
_P = 2**255 - 19
_ELL = 2**252 + 27742317777372353535851937790883648493


def _inv(x: int) -> int:
    return pow(x, _P - 2, _P)


_D = -121665 * _inv(121666) % _P
_I = pow(2, (_P - 1) // 4, _P)


def _recover_x(y: int, sign: int) -> int | None:
    if y >= _P:
        return None
    x2 = (y * y - 1) * _inv(_D * y * y + 1) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = (x * _I) % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_BY = 4 * _inv(5) % _P
_BX = _recover_x(_BY, 0)
assert _BX is not None
# Extended projective coordinates: (X, Y, Z, T) where x = X/Z, y = Y/Z, T = XY/Z
_B_EXT = (_BX, _BY, 1, (_BX * _BY) % _P)


def _point_add(
    p1: tuple[int, int, int, int], p2: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Extended twisted Edwards point addition (Hisil et al., Asiacrypt 2008)."""
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    a = (y1 - x1) * (y2 - x2) % _P
    b = (y1 + x1) * (y2 + x2) % _P
    c = 2 * _D * t1 * t2 % _P
    d = 2 * z1 * z2 % _P
    e = (b - a) % _P
    f = (d - c) % _P
    g = (d + c) % _P
    h = (b + a) % _P
    x3 = (e * f) % _P
    y3 = (g * h) % _P
    t3 = (e * h) % _P
    z3 = (f * g) % _P
    return (x3, y3, z3, t3)


def _point_double(p: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Extended twisted Edwards point doubling."""
    x1, y1, z1, _ = p
    a = (x1 * x1) % _P
    b = (y1 * y1) % _P
    c = (2 * z1 * z1) % _P
    h = (a + b) % _P
    e = (h - pow(x1 + y1, 2, _P)) % _P
    g = (a - b) % _P
    f = (c + g) % _P
    x3 = (e * f) % _P
    y3 = (g * h) % _P
    t3 = (e * h) % _P
    z3 = (f * g) % _P
    return (x3, y3, z3, t3)


def _point_to_affine(p: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, z, _ = p
    z_inv = _inv(z)
    return ((x * z_inv) % _P, (y * z_inv) % _P)


def _scalar_mul(p: tuple[int, int, int, int], e: int) -> tuple[int, int, int, int]:
    res = (0, 1, 1, 0)
    base = p
    while e > 0:
        if e & 1:
            res = _point_add(res, base)
        base = _point_double(base)
        e >>= 1
    return res


def _encodepoint(p_ext: tuple[int, int, int, int]) -> bytes:
    x, y = _point_to_affine(p_ext)
    b = bytearray(y.to_bytes(32, "little"))
    b[31] |= (x & 1) << 7
    return bytes(b)


def _decodepoint(s: bytes) -> tuple[int, int, int, int] | None:
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign = (y >> 255) & 1
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, (x * y) % _P)


def _publickey_pure(sk: bytes) -> bytes:
    h = hashlib.sha512(sk).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    pt_ext = _scalar_mul(_B_EXT, a)
    return _encodepoint(pt_ext)


def _sign_pure(m: bytes, sk: bytes, pk: bytes) -> bytes:
    h = hashlib.sha512(sk).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    r = int.from_bytes(hashlib.sha512(h[32:] + m).digest(), "little") % _ELL
    pt_r = _scalar_mul(_B_EXT, r)
    r_bytes = _encodepoint(pt_r)
    k = int.from_bytes(hashlib.sha512(r_bytes + pk + m).digest(), "little") % _ELL
    s = (r + k * a) % _ELL
    return r_bytes + s.to_bytes(32, "little")


def _verify_pure(sig: bytes, m: bytes, pk: bytes) -> bool:
    if len(sig) != 64 or len(pk) != 32:
        return False
    r_bytes = sig[:32]
    s = int.from_bytes(sig[32:], "little")
    if s >= _ELL:
        return False
    pt_r = _decodepoint(r_bytes)
    pt_a = _decodepoint(pk)
    if pt_r is None or pt_a is None:
        return False
    k = int.from_bytes(hashlib.sha512(r_bytes + pk + m).digest(), "little") % _ELL
    sb = _scalar_mul(_B_EXT, s)
    ka = _scalar_mul(pt_a, k)
    r_plus_ka = _point_add(pt_r, ka)
    return _encodepoint(sb) == _encodepoint(r_plus_ka)


# --- Public High-Level Interface ---


def generate_keypair() -> tuple[bytes, bytes]:
    """
    Generate a new Ed25519 keypair.

    Returns:
        Tuple of (private_key_bytes, public_key_bytes), each 32 bytes.
    """
    if _HAS_CRYPTOGRAPHY and _crypto_ed25519 is not None:
        priv = _crypto_ed25519.Ed25519PrivateKey.generate()
        priv_bytes = priv.private_bytes_raw()
        pub_bytes = priv.public_key().public_bytes_raw()
        return priv_bytes, pub_bytes

    sk = secrets.token_bytes(32)
    pk = _publickey_pure(sk)
    return sk, pk


def generate_keypair_hex() -> tuple[str, str]:
    """
    Generate a new Ed25519 keypair formatted as hex strings.

    Returns:
        Tuple of (private_key_hex, public_key_hex).
    """
    sk, pk = generate_keypair()
    return sk.hex(), pk.hex()


def public_key_from_private_key(private_key: bytes | str) -> bytes:
    """
    Derive the 32-byte public key from a given 32-byte private key.

    Args:
        private_key: 32-byte bytes or 64-character hex string.

    Returns:
        32-byte public key.
    """
    sk = bytes.fromhex(private_key) if isinstance(private_key, str) else private_key
    if len(sk) != 32:
        raise ValueError(f"Ed25519 private key must be 32 bytes, got {len(sk)}")

    if _HAS_CRYPTOGRAPHY and _crypto_ed25519 is not None:
        priv = _crypto_ed25519.Ed25519PrivateKey.from_private_bytes(sk)
        return bytes(priv.public_key().public_bytes_raw())

    return _publickey_pure(sk)


def sign(message: bytes, private_key: bytes | str) -> bytes:
    """
    Compute a 64-byte Ed25519 digital signature over the given message bytes.

    Args:
        message: Raw message bytes to sign.
        private_key: 32-byte bytes or 64-character hex string.

    Returns:
        64-byte signature.
    """
    sk = bytes.fromhex(private_key) if isinstance(private_key, str) else private_key
    if len(sk) != 32:
        raise ValueError(f"Ed25519 private key must be 32 bytes, got {len(sk)}")

    if _HAS_CRYPTOGRAPHY and _crypto_ed25519 is not None:
        priv = _crypto_ed25519.Ed25519PrivateKey.from_private_bytes(sk)
        return bytes(priv.sign(message))

    pk = _publickey_pure(sk)
    return _sign_pure(message, sk, pk)


def verify(signature: bytes | str, message: bytes, public_key: bytes | str) -> bool:
    """
    Verify an Ed25519 digital signature using only the public key.

    Args:
        signature: 64-byte signature bytes or 128-character hex string.
        message: Raw message bytes originally signed.
        public_key: 32-byte public key bytes or 64-character hex string.

    Returns:
        True if valid; False if forged, corrupted, or key mismatch.
    """
    try:
        sig = bytes.fromhex(signature) if isinstance(signature, str) else signature
        pk = bytes.fromhex(public_key) if isinstance(public_key, str) else public_key
    except (ValueError, TypeError):
        return False

    if len(sig) != 64 or len(pk) != 32:
        return False

    if _HAS_CRYPTOGRAPHY and _crypto_ed25519 is not None:
        try:
            pub = _crypto_ed25519.Ed25519PublicKey.from_public_bytes(pk)
            pub.verify(sig, message)
            return True
        except Exception:
            return False

    return _verify_pure(sig, message, pk)
