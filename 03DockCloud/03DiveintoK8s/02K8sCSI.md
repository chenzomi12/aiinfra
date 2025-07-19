# Kubernetes 容器持久化存储
开发者视角：只需声明存储需求，无需关注底层实现。

运维视角：自由更换存储后端（NFS/云盘/分布式存储）不影响业务。

核心价值：实现应用与基础设施的关注点分离。

## 核心概念与架构
### 存储抽象层
```mermaid
A[Pod] --> B[Volume]
B --> C[PersistentVolumeClaim]
C --> D[PersistentVolume]
D --> E[StorageClass]
E --> F[实际存储系统]
```


### 核心组件关系
| 组件 | 角色 | 生命周期 | 创建者 |
|------|------|----------|----------|
| PersistentVolume (PV) | 集群存储资源 | 独立于 Pod | 管理员/StorageClass |
| PersistentVolumeClaim (PVC) | 用户存储请求 | 与应用 Pod 关联 | 开发者 |
| StorageClass (SC) | 存储动态供应模板 | 集群级别长期存在 | 管理员 |


## 存储类型详解
### 卷类型比较
| 类型 | 特点 | 适用场景 | 示例 |
|------|------|----------|----------|
| 本地存储 | 高性能，节点绑定 | 临时缓存 | hostPath, emptyDir |
| 网络存储 | 跨节点共享 | 数据库/共享文件 | NFS, CephFS |
| 云存储 | 托管服务，高可用 | 云原生应用 | AWS EBS, GCP PD |
| 分布式存储 | 可扩展性强 | 大数据平台 | GlusterFS, Ceph RBD |

### 访问模式
| 类型 | 特点 | 适用场景 | 
|------|------|----------|
| ReadWriteOnce (RWO) | 单节点读写 | 块存储(EBS) | 
| ReadOnlyMany (ROX) | 多节点只读	 | 文件存储(NFS) |
| ReadWriteMany (RWX)	 | 多节点读写	 | 分布式存储(CephFS) | 

## Volume
Volume（存储卷） 是用于在容器之间或容器重启后持久化数据的抽象层。与 Docker 容器内的临时存储不同，k8s Volume 的生命周期与 Pod 绑定，而非单个容器，确保数据在容器重启、重建或跨容器共享时不丢失。

### 核心作用
- 数据持久化：容器重启后数据不丢失（容器本身的文件系统是临时的）。
- 跨容器共享：同一 Pod 内的多个容器可通过 Volume 共享数据。
- 集成外部存储：对接各种外部存储系统（如本地磁盘、云存储、分布式存储等）。
- 配置管理：通过 Volume 向容器注入配置文件（如 ConfigMap、Secret）。

### 临时卷类型
| 类型 | 说明 | 场景示例 | 
|------|------|----------|
| emptyDir | Pod 创建时自动创建的空目录，存储在节点本地（内存或磁盘） | 容器间临时数据共享、缓存 | 
| configMap | 将 ConfigMap 中的配置数据挂载为文件	 | 应用配置注入 |
| secret	 | 类似 ConfigMap，但用于存储敏感数据（如密码、证书），数据会 base64 编码	 | 数据库密码、API 密钥挂载 | 
| downwardAPI	 | 将 Pod 或容器的元数据（如名称、IP）挂载为文件	 | 应用获取自身运行时信息 | 

### 持久化存储类（跨 Pod 生命周期）
| 类型                          | 说明                                                                 | 场景示例                     |
|-------------------------------|----------------------------------------------------------------------|------------------------------|
| **persistentVolumeClaim (PVC)** | 通过 PVC 动态绑定 PersistentVolume (PV)，抽象存储细节                | 数据库数据持久化             |
| **hostPath**                  | 将节点本地磁盘路径直接挂载到 Pod (不推荐在生产环境使用)              | 开发测试、节点级日志存储     |
| **nfs**                       | 挂载 NFS (网络文件系统) 共享目录                                     | 跨节点数据共享               |
| **云厂商存储**                | 如 awsElasticBlockStore (AWS)、gcePersistentDisk (GCP)、azureDisk (Azure) | 云环境中持久化存储           |
| **分布式存储**                | 如 cephfs、glusterfs、rook (基于 Ceph)                              | 大规模分布式存储需求         |

### 特殊用途卷类型csi
CSI（Container Storage Interface）是容器存储接口的标准，允许存储供应商编写插件来支持其存储系统。CSI 卷类型允许 Pod 使用任何符合 CSI 规范的存储驱动程序。
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: csi-pod
spec:
  containers:
  - name: test-container
    image: nginx:1.20
    volumeMounts:
    - mountPath: /data
      name: csi-volume
  volumes:
  - name: csi-volume
    csi:
      driver: my-csi-driver
      volumeAttributes:
        storage.kubernetes.io/csiProvisionerIdentity: my-provisioner
```

## 持久化卷（Persistent Volume）
### 核心概念
持久化卷实现存储资源的统一管理。
- PersistentVolume：PV 具有独立于 Pod 的生命周期，封装了底层存储实现的具体细节。
- PersistentVolumeClaim：类似于 Pod 消耗节点资源，PVC 消耗 PV 资源。用户通过 PVC 请求特定大小和访问模式的存储，无需了解底层存储实现。
- StorageClass：支持动态配置、不同的服务质量级别，并可定义配置参数和回收策略。

### 关键阶段说明
| 阶段           | 触发条件       | 系统行为             | 持续时间          |
|----------------|----------------|----------------------|-------------------|
| Provisioning   | PVC创建        | 动态分配存储资源     | 秒级~分钟级       |
| Binding        | PVC匹配PV      | 建立绑定关系         | 瞬时完成          |
| Using          | Pod挂载        | 数据读写操作         | 应用运行期        |
| Releasing      | PVC删除        | 解除PV绑定           | 瞬时完成          |
| Reclaiming     | PV释放         | 执行回收策略         | 依赖策略类型      |
| Recycling      | 回收完成       | 等待重新绑定         | 无限期            |

### 动态生命周期管理
StorageClass 控制机制
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: managed-premium
provisioner: kubernetes.io/azure-disk
reclaimPolicy: Delete # 核心回收策略
allowVolumeExpansion: true # 允许扩容
volumeBindingMode: WaitForFirstConsumer # 延迟绑定
parameters:
  skuname: Premium_LRS
```

延迟绑定(WaitForFirstConsumer)
```flowchart TD
    A[创建PVC] --> B[等待Pod调度]
    B --> C{确定节点}
    C --> D[在目标节点所在区创建PV]
    D --> E[绑定PVC-PV]
```

卷扩容流程
```sequenceDiagram
    User->>PVC: kubectl edit pvc size=30Gi
    PVC->>StorageClass: 检查allowVolumeExpansion
    StorageClass->>Cloud-Plugin: 调用扩容API
    Cloud-Plugin->>Storage: 扩展卷容量
    Storage-->>Node: 通知文件系统扩容
    Node->>Pod: 在线扩容无需重启
```

## 工作流程解析
静态工作流程
```mermaid
sequenceDiagram
    管理员->>集群： 创建PV (my-pv)
    开发者->>集群： 创建PVC (my-pvc)
    集群->>集群： 绑定PV和PVC
    开发者->>集群： 创建Pod使用PVC
    Pod->>PV： 挂载存储
```

动态工作流程
```sequenceDiagram
    开发者->>集群： 创建PVC (指定StorageClass)
    集群->>StorageClass： 请求创建PV
    StorageClass->>云存储： 调用API创建卷
    云存储-->>集群： 返回新PV
    集群->>PVC： 自动绑定
```

## 关键配置详解
### StorageClass示例
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  fsType: ext4
  iops: "10000"
  throughput: "250"
reclaimPolicy: Delete
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

### PV/PVC 状态机
```yaml
stateDiagram-v2
    PV: PersistentVolume
    state PV {
        [*] --> Available
        Available --> Bound
        Bound --> Released
        Released --> Available
        Released --> Failed
    }
    
    PVC: PersistentVolumeClaim
    state PVC {
        [*] --> Pending
        Pending --> Bound
        Bound --> Lost
    }
```

## 高级特性
### 数据保护机制
卷快照
```yaml
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: db-snapshot
spec:
  volumeSnapshotClassName: csi-aws-vsc
  source:
    persistentVolumeClaimName: mysql-pvc

```
克隆技术：直接从快照创建新卷
数据迁移：跨集群/云厂商卷迁移

### 存储扩展
```bash
kubectl edit pvc my-pvc # 修改storage请求大小
```

拓扑感知：使用 volumeBindingMode: WaitForFirstConsumer
存储配额：限制命名空间存储用量


