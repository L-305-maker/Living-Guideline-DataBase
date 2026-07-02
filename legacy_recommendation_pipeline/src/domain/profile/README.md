# domain/profile

这里定义指南画像相关实体，用于描述指南来源、类型、适用范围等上下文。

## 数据流动

```text
cleaned record / guideline profile
  -> extraction/versioning
  -> normalized_payload.profile_context
```

画像信息通常不直接决定发布，但会帮助解释推荐来自什么类型的指南和上下文。
