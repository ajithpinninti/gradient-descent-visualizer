/**
 * Tiny test harness — no dependencies.
 * Run: node tests/harness.js
 */

export function assert(condition, message) {
  if (!condition) {
    throw new Error(message || "Assertion failed");
  }
}

export function assertClose(actual, expected, tolerance = 1e-5, message = "") {
  const diff = Math.abs(actual - expected);
  if (diff > tolerance) {
    throw new Error(
      `${message} Expected ${expected}, got ${actual} (diff ${diff})`
    );
  }
}

export function assertArrayClose(actual, expected, tolerance = 1e-5, message = "") {
  assert(actual.length === expected.length, `${message} length mismatch`);
  for (let i = 0; i < actual.length; i++) {
    assertClose(actual[i], expected[i], tolerance, `${message}[${i}]`);
  }
}

export async function runSuite(name, tests) {
  let passed = 0;
  let failed = 0;
  const failures = [];

  for (const [testName, fn] of Object.entries(tests)) {
    try {
      await fn();
      console.log(`  ✓ ${testName}`);
      passed++;
    } catch (err) {
      console.log(`  ✗ ${testName}`);
      failures.push({ testName, err });
      failed++;
    }
  }

  console.log(`\n${name}: ${passed} passed, ${failed} failed`);
  if (failures.length) {
    for (const { testName, err } of failures) {
      console.error(`  ${testName}: ${err.message}`);
    }
    process.exit(1);
  }
}
