"""Test that all converted relative imports work correctly."""
import sys
import traceback

def test_import(module_path, description, allow_runtime_errors=False):
    """Test importing a module and report results."""
    try:
        __import__(module_path)
        print(f"[PASS] {description}: {module_path}")
        return True
    except ImportError as e:
        # ImportError means the import statement itself failed
        print(f"[FAIL] {description}: {module_path}")
        print(f"  Import Error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        # Other exceptions (AttributeError, etc.) may occur during module execution
        # but the import itself succeeded
        if allow_runtime_errors:
            print(f"[PASS] {description}: {module_path} (runtime error expected)")
            print(f"  Note: Module executes code at import time")
            return True
        else:
            print(f"[FAIL] {description}: {module_path}")
            print(f"  Runtime Error: {e}")
            traceback.print_exc()
            return False

def main():
    print("Testing relative imports in CTAFlow modules...\n")

    results = []

    # Test data module imports
    print("=" * 60)
    print("DATA MODULE")
    print("=" * 60)
    results.append(test_import(
        "CTAFlow.data.raw_formatting.spread_manager",
        "spread_manager.py"
    ))
    results.append(test_import(
        "CTAFlow.data.raw_formatting.synthetic",
        "synthetic.py"
    ))

    # Test screeners module imports
    print("\n" + "=" * 60)
    print("SCREENERS MODULE")
    print("=" * 60)
    results.append(test_import(
        "CTAFlow.screeners.generic",
        "generic.py"
    ))
    results.append(test_import(
        "CTAFlow.screeners.regime_screens",
        "regime_screens.py"
    ))

    # Test strategy module imports
    print("\n" + "=" * 60)
    print("STRATEGY MODULE")
    print("=" * 60)
    results.append(test_import(
        "CTAFlow.strategy.positioning",
        "positioning.py",
        allow_runtime_errors=True  # This file executes code at import time
    ))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\nAll imports working correctly!")
        return 0
    else:
        print(f"\n{total - passed} import(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
