# Hendrycks MATH Dataset — Metric Analysis

**Source:** `EleutherAI/hendrycks_math` | **Splits:** Train / Test | **Subjects:** 7 | **Difficulty Levels:** 5 (+ 1 anomaly in Geometry)

---

## Per-Subject Split Counts by Difficulty

| Level | Algebra | | Counting & Probability | | Geometry | | Intermediate Algebra | | Number Theory | | Prealgebra | | Precalculus | |
|:------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | **Train** | **Test** | **Train** | **Test** | **Train** | **Test** | **Train** | **Test** | **Train** | **Test** | **Train** | **Test** | **Train** | **Test** |
| Level 1 | 178 | 135 | 50 | 39 | 41 | 38 | 58 | 52 | 47 | 30 | 126 | 86 | 64 | 57 |
| Level 2 | 340 | 201 | 119 | 101 | 100 | 82 | 167 | 128 | 131 | 92 | 329 | 177 | 162 | 113 |
| Level 3 | 392 | 261 | 160 | 100 | 129 | 102 | 301 | 195 | 191 | 122 | 251 | 224 | 168 | 127 |
| Level 4 | 398 | 283 | 166 | 111 | 177 | 125 | 340 | 248 | 187 | 142 | 247 | 191 | 175 | 114 |
| Level 5 | 436 | 307 | 276 | 123 | 421 | 132 | 429 | 280 | 313 | 154 | 252 | 193 | 177 | 135 |
| Level ? | — | — | — | — | 2 | 0 | — | — | — | — | — | — | — | — |
| **Total** | **1,744** | **1,187** | **771** | **474** | **870** | **479** | **1,295** | **903** | **869** | **540** | **1,205** | **871** | **746** | **546** |

> **Note:** `Level ?` is a data anomaly exclusive to Geometry — 2 mislabeled entries in the training split with no corresponding test entries.

---

## Summary Totals

| Subject | Difficulties | Train | Test | Total |
|:--------|:------------:|------:|-----:|------:|
| Algebra | 5 | 1,744 | 1,187 | 2,931 |
| Counting & Probability | 5 | 771 | 474 | 1,245 |
| Geometry | 6 | 870 | 479 | 1,349 |
| Intermediate Algebra | 5 | 1,295 | 903 | 2,198 |
| Number Theory | 5 | 869 | 540 | 1,409 |
| Prealgebra | 5 | 1,205 | 871 | 2,076 |
| Precalculus | 5 | 746 | 546 | 1,292 |
| **Grand Total** | | **7,500** | **5,000** | **12,500** |
