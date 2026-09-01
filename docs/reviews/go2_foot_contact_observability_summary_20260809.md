# Go2 foot-contact observability diagnostic

- Decision: `INCONCLUSIVE_DO_NOT_TRAIN`
- Analysis status: `coverage_inconclusive`
- Manifest SHA256: `bbcdb4eb754e54d26186762e1a0353e542ac09890d98df5ce61a425349327c6b`
- Contract SHA256: `cc8c0dc49f6abec8d1e99d031d8742ec265869a4bd0aca14fab947f773492517`
- Training changed: `false`; `learn()` called: `false`

## Decision reasons

- coverage_failed
- clean|vx=0.3|H=10:contact_chatter_above_0.10
- clean|vx=0.3|H=10:catastrophic_failure:positive_anchors_below_200
- clean|vx=0.5|H=10:contact_chatter_above_0.10
- clean|vx=0.5|H=10:catastrophic_failure:positive_anchors_below_200
- randomized|vx=0.3|H=10:contact_chatter_above_0.10
- randomized|vx=0.3|H=10:catastrophic_failure:positive_anchors_below_200
- randomized|vx=0.5|H=10:contact_chatter_above_0.10
- randomized|vx=0.5|H=10:catastrophic_failure:positive_anchors_below_200
- clean|vx=0.3|H=25:contact_chatter_above_0.10
- clean|vx=0.3|H=25:catastrophic_failure:positive_anchors_below_200
- clean|vx=0.5|H=25:contact_chatter_above_0.10
- randomized|vx=0.3|H=25:contact_chatter_above_0.10
- randomized|vx=0.3|H=25:catastrophic_failure:positive_anchors_below_200
- randomized|vx=0.5|H=25:contact_chatter_above_0.10
- clean|vx=0.3|H=50:contact_chatter_above_0.10
- clean|vx=0.5|H=50:contact_chatter_above_0.10
- randomized|vx=0.3|H=50:contact_chatter_above_0.10
- randomized|vx=0.5|H=50:contact_chatter_above_0.10

## Coverage

| profile / speed / H | pass | clusters | slip + | unexpected + | failure + | progress rows | rays |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean|vx=0.3|H=10 | False | 24 | 6163 | 3919 | 54 | 21205 | 1.0000 |
| clean|vx=0.5|H=10 | False | 24 | 5447 | 3075 | 101 | 9497 | 0.9999 |
| randomized|vx=0.3|H=10 | False | 24 | 7062 | 5809 | 52 | 21818 | 1.0000 |
| randomized|vx=0.5|H=10 | False | 24 | 6418 | 2840 | 92 | 10316 | 1.0000 |
| clean|vx=0.3|H=25 | False | 24 | 8632 | 7658 | 135 | 20989 | 1.0000 |
| clean|vx=0.5|H=25 | False | 24 | 6847 | 4921 | 254 | 9281 | 0.9999 |
| randomized|vx=0.3|H=25 | False | 24 | 10064 | 11452 | 130 | 21602 | 1.0000 |
| randomized|vx=0.5|H=25 | False | 24 | 8201 | 5007 | 230 | 10100 | 1.0000 |
| clean|vx=0.3|H=50 | False | 24 | 9383 | 9600 | 270 | 20629 | 1.0000 |
| clean|vx=0.5|H=50 | False | 24 | 7348 | 6004 | 509 | 8921 | 0.9999 |
| randomized|vx=0.3|H=50 | False | 24 | 12173 | 15560 | 260 | 21242 | 1.0000 |
| randomized|vx=0.5|H=50 | False | 24 | 8985 | 6527 | 460 | 9740 | 1.0000 |

## Interpretation

The evidence is inconclusive under the frozen gate. Do not start the 238D teacher training from this result.
