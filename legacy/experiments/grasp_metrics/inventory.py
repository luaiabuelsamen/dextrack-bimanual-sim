"""What this machine can build. Run it to see the embodiment registry."""
from handsim.embodiment import inventory


def main():
    rows = inventory()
    print(f"{'kind':6}{'key':12}{'nq':>4}{'nu':>4}  {'status':8} note")
    for kind, k, nq, nu, note, st in rows:
        print(f"{kind:6}{k:12}{nq:>4}{nu:>4}  {st:8} {note}")
    bad = [k for _, k, _, _, _, st in rows if st != "ok"]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} load" +
          (f"; failing: {bad}" if bad else ""))


if __name__ == "__main__":
    main()
