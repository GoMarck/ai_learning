# Mooncake Transfer Engine

## 1. 概述

Mooncake Transfer Engine(TE) 是一个高性能、零拷贝的的数据传输库，它提供了以下两个核心概念：
- **Segment**：代表一个连续地址空间，可远端读写；它可以是由 DRAM 或 VRAM 提供的非持久性存储（称为 RAM Segment），也可以是由 NVMeof 提供的持久性存储（称为 NVMeof Segment）。
- **BtachTransfer**：封装了操作请求，专门负责在一个 Segment 中的一组不连续数据空间与另一组 Segment 中的相应空间之间同步数据，支持双向读写，因此其作用类似于异步且更灵活的 AllScatter/AllGather。

![](images/transfer-engine.png)

## 2. Segment

## 3. BatchTransfer

## 4. Topology

## 5. API

## 6. Example

