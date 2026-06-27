# domain/update

这里定义推荐更新事件实体。

## 数据流动

```text
old RecommendationVersion + new RecommendationVersion
  -> version_diff
  -> UpdateLog
  -> publish / storage
```

`UpdateLog` 用于回答“这条推荐和上一版相比发生了什么变化”。
