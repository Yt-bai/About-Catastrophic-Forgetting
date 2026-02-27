## 1️⃣ 什么是灾难性遗忘？

**假设：**

- 先训练 Task A  
- 再训练 Task B  

**结果：**

- Task B 准确率高  
- Task A 准确率几乎归零  

**原因：**

梯度更新覆盖了旧知识。


## 持续学习的核心问题

持续学习（Continual Learning）的本质在于解决：

👉 **Stability–Plasticity Tradeoff**

---

你希望模型同时具备：

- **Plasticity**：能够学习新任务  
- **Stability**：能够保留旧任务知识  

---

但这两种能力在优化过程中是天然冲突的。

