# VLA-Adapter 复现记录（基于 LIBERO）
## 1. 简介

本仓库用于记录在广工服务器上复现 **VLA-Adapter**（LIBERO 任务）全过程，包含：

- 环境搭建（conda + 离线 pip + mujoco/robosuite）
- 数据集路径配置（本地 `modified_libero_rlds`）
- 训练脚本与关键启动参数
- 推理 / 评估脚本
- 常见报错与修复办法（本次复现真实发生过的）
- 
## 2. 环境搭建

### 2.1 创建虚拟环境

```bash
conda create -n adapter python=3.10 -y
conda activate adapter
