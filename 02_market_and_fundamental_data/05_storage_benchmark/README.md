## Efficient data storage with pandas

[storage_benchmark](storage_benchmark.ipynb)에서는 데이터 포맷의 efficiency와 performance를 비교한다.

- CSV: 수치형 데이터 저장에 좋음
- HDF5: 무난함, PyTable 라이브러리 통해서 사용 가능 - hadoop이랑 호환 좋음
- Parquet: text, 수치 등 섞인 데이터 저장용으로 많이 사용, pyarrow 라이브러리 통해서 사용 가능 - spark랑 호환 좋음