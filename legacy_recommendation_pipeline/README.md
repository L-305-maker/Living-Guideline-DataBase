# legacy_recommendation_pipeline

这个目录存放旧的 Recommendation/PICO/GRADE 抽取、审核、发布和旧测试脚本。

这些文件当前不再参与新的“指南清洗切分并建库，为循证医学 Agent 提供证据”任务，但保留在这里用于历史参考、回滚或以后单独恢复。

迁移后的 active 代码集中在：

- `src/pipeline/cleaning/`
- `src/pipeline/orchestration/`
- `src/common/mcp_server.py`
- `project/`

如果以后需要恢复旧推荐抽取链路，建议从本目录单独开分支整理，不要直接混回 active pipeline。
