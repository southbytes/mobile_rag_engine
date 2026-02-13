# Testing Tiers

The 'mobile_rag_engine' test is divided into three layers according to the purpose.

## 1) Unit (`test/unit`)

- Objective: To verify pure Dart logic (minimize environmental dependence)
- Features: Fast and Reproducible
- Run:

```bash
./scripts/test_ci.sh unit
```

## 2) Native (`test/native`)

- Purpose: Verification of Rust/FFI path inclusion
- Features: Existence of environmental dependence (primarily running on macOS runner)
- Run:

```bash
./scripts/test_ci.sh native
```

## 3) Integration (`integration_test`)

- Purpose: App Path Unit Integration Verification
- Feature: If no test, automatic skip from CI
- Run:

```bash
./scripts/test_ci.sh integration
```

## CI Gate Mapping

- Ubuntu: `analyze + unit`
- macOS: `native`
- macOS: 'integration' (if present)
- Platform build smoke: `android`, `ios`, `macos`