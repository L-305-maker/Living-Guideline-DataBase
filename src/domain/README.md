# domain 模块

`domain/` 定义项目里的核心业务对象。它回答“数据长什么样”，不负责“数据怎么被抽取、怎么写文件、怎么入库”。

## 数据如何流动

```text
pipeline/extraction
  -> 构造 domain 对象
  -> to_dict()
  -> JSONL 产物
  -> storage 行映射 / 发布包
```

## 子目录

| 子目录 | 作用 |
| --- | --- |
| `origin/` | 原始来源实体，如 SourceRecord、SourceBlock、Guideline、Paper |
| `candidate/` | 抽取候选实体，如 RecommendationCandidate、GradeCandidate、ModelTrace |
| `knowledge/` | 更接近正式知识库的实体，如 PicoQuestion、EvidenceItem、RecommendationVersion |
| `profile/` | 指南画像和上下文信息 |
| `update/` | 推荐更新事件，如 UpdateLog |
| `common/` | 枚举、ID、序列化基础能力 |

## 关键边界

```text
Candidate
  = 抽取出来、还需要复核的候选

Knowledge
  = 更接近正式知识库的数据结构，但仍可能需要 publish gate

Storage row
  = 入库前按 PostgreSQL schema 整理后的行
```

维护时不要把抽取规则写进领域模型。领域模型应该尽量保持“描述数据”，而不是“决定数据”。
