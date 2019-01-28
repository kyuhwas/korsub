# 한국어 subword embedding 과 관련된 데이터

## LR_surface_9tags

한국어의 어절 구조를 L + [R] 로 정리한 뒤, L 과 R 에 대한 표현형의 (L, R) count 를 정리한 데이터로, L 과 R 이 표현형이기 때문에 용언이 활용된 형태 그대로 보존됩니다.

```
표현형: 했다 -> 했/Verb + 다
원형:  했다 -> 하/Verb + 았다
```

이 데이터는 아래의 함수를 통하여 이용할 수 있습니다.

```python
from korsub_data import load_lr_surface_9tags

X, idx_to_l, idx_to_r, idx_to_ltag, idx_to_lmorph = load_lr_surface_9tags()
```