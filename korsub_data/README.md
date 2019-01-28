# 한국어 subword embedding 과 관련된 데이터

## LR surface 9tags

한국어의 어절 구조를 L + [R] 표현형으로 정리한 뒤 (L, R) count 를 계산한 데이터로, L 과 R 이 표현형이기 때문에 용언이 활용된 형태 그대로 보존됩니다.

```
표현형: 했다 -> 했/Verb + 다
원형:  했다 -> 하/Verb + 았다
```

이 데이터는 아래의 함수를 통하여 이용할 수 있습니다.

```python
from korsub_data import load_lr_surface_9tags

X, idx_to_l, idx_to_r, idx_to_ltag, idx_to_lmorph = load_lr_surface_9tags()
```

## LR surface noun

한국어의 어절 구조를 L + [R] 표현형으로 정리한 뒤, 명사 추출용으로 정제한 데이터로, L 이 명사일 경우 label `1`, 명사가 아닐 경우 label `-1` 을 지닙니다.

```python
from korsub_data import load_lr_surface_noun

X, idx_to_l, idx_to_r, idx_to_ltag, idx_to_lmorph = load_lr_surface_noun()
```