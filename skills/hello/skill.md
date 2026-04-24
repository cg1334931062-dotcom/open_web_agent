---
name: hello
description: 简单问候，返回欢迎信息和当前时间
parameters:
  type: object
  properties:
    name:
      type: string
      description: 要问候的名字
      default: "用户"
  required: []
run:
  type: bash
  command: 'echo "你好，{name}！当前时间：$(date +%Y-%m-%d\ %H:%M:%S)"'
---

# Hello Skill

简单的问候技能，返回欢迎信息和当前时间。
