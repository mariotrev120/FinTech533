"""
TWS connectivity diagnostic.

Probes: (1) TCP reachability, (2) raw API handshake byte-by-byte with a long
read timeout, (3) ib_async connection with several client IDs, (4) prints the
current WSL IP and the TWS host IP we're targeting.

Run from WSL:
    /home/mht120/projects/FinTech533/Trading/.venv/bin/python tws_debug.py
"""
from __future__ import annotations
import socket
import struct
import subprocess
import time

HOST = "172.29.208.1"
PORT = 7497


def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception as e:
        return f"ERR: {e}"


def section(t):
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


section("1. Network context")
print(f"  WSL IP (add THIS to TWS trusted IPs):  {sh('hostname -I | awk \"{print $1}\"')}")
print(f"  WSL gateway (= Windows host):           {sh('ip route show default | awk \"{print $3}\"')}")
print(f"  Target TWS:                             {HOST}:{PORT}")


section("2. TCP reachability")
s = socket.socket()
s.settimeout(3)
t0 = time.time()
try:
    s.connect((HOST, PORT))
    print(f"  TCP connect: OK in {1000*(time.time()-t0):.0f} ms")
    s.close()
except Exception as e:
    print(f"  TCP connect FAILED: {e}")
    raise SystemExit(1)


section("3. Raw IBKR API handshake")
s = socket.socket()
s.settimeout(10)
s.connect((HOST, PORT))
body = b"v100..187"
msg = b"API\0" + struct.pack("!I", len(body)) + body
s.sendall(msg)
print(f"  sent {len(msg)} bytes (API-prefix + length-prefixed 'v100..187')")
print(f"  waiting up to 10s for response...")
try:
    data = s.recv(4096)
    print(f"  RECEIVED {len(data)} bytes: {data!r}")
    if data:
        print("  -> TWS accepted the handshake. Any subsequent code failures are NOT permissions.")
    else:
        print("  -> TWS closed the socket without replying. Classic permission rejection.")
except socket.timeout:
    print("  -> TIMEOUT: TWS held the socket but sent nothing. Likely waiting on a user-accept dialog.")
except ConnectionResetError:
    print("  -> RST from TWS. Usually trusted-IP mismatch or TWS not logged in.")
except Exception as e:
    print(f"  -> {type(e).__name__}: {e}")
finally:
    s.close()


section("4. ib_async attempts with several client IDs")
try:
    from ib_async import IB
except ImportError:
    print("  ib_async not installed, skipping")
else:
    for cid in [0, 1, 100, 2000]:
        ib = IB()
        t0 = time.time()
        try:
            ib.connect(HOST, PORT, clientId=cid, timeout=8)
            print(f"  client_id={cid}: CONNECTED in {1000*(time.time()-t0):.0f} ms  accounts={ib.managedAccounts()}")
            ib.disconnect()
            break
        except Exception as e:
            print(f"  client_id={cid}: {type(e).__name__}: {str(e)[:120]}")


section("5. Things to verify in TWS now")
print("""
  [ ] TWS window is fully logged in (title bar shows 'Paper Trading - DUO566952')
  [ ] No dialog popup waiting for a click (alt-tab through every TWS window)
  [ ] File -> Global Configuration -> API -> Settings:
        [x] Enable ActiveX and Socket Clients
        [ ] Allow connections from localhost only  (UNCHECKED)
        [ ] Read-Only API                          (either)
        Socket port: 7497
        Trusted IP Addresses: must include the WSL IP printed in section 1
  [ ] If you changed ANY of the above: click Apply, then File -> Close TWS,
      wait 10 seconds, and relaunch. Settings only take effect after restart.
  [ ] While reconnecting, tail the TWS API log in Notepad:
        C:\\Users\\mtrev0001\\Jts\\api.YYYYMMDD.log
      That log will state the exact reason for each rejection.
""")
