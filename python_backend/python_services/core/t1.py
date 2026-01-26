"""
后台任务管理器
"""
import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from concurrent.futures import Future

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"  # 等待执行
    RUNNING = "running"  # 正在执行
    COMPLETED = "completed"  # 执行完成
    FAILED = "failed"  # 执行失败
    STOPPED = "stopped"  # 已停止


class ScheduleType(Enum):
    """调度类型"""
    INTERVAL = "interval"  # 固定间隔执行
    CRON = "cron"  # cron表达式
    DELAYED = "delayed"  # 延迟执行一次


@dataclass
class TaskConfig:
    """任务配置"""
    name: str  # 任务名称
    func: Callable  # 任务函数
    schedule_type: ScheduleType  # 调度类型
    interval: Optional[timedelta] = None  # 执行间隔（仅INTERVAL类型）
    cron_expression: Optional[str] = None  # cron表达式（仅CRON类型）
    delay: Optional[timedelta] = None  # 延迟时间（仅DELAYED类型）
    max_retries: int = 3  # 最大重试次数
    retry_delay: timedelta = field(default_factory=lambda: timedelta(seconds=10))
    args: tuple = ()  # 函数参数
    kwargs: Dict[str, Any] = field(default_factory=dict)  # 函数关键字参数
    enabled: bool = True  # 是否启用


@dataclass
class TaskResult:
    """任务执行结果"""
    task_name: str
    status: TaskStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    error: Optional[Exception] = None
    result: Any = None
    retry_count: int = 0


class BackgroundTaskManager:
    """
    后台任务管理器

    使用示例：
    ```python
    # 创建任务管理器
    task_manager = BackgroundTaskManager()

    # 添加周期性任务
    task_manager.add_interval_task(
        name="save_faiss",
        func=faiss_index.save_local,
        interval=timedelta(minutes=30),
        args=("faiss_index",),
        kwargs={"index_name": "my_index"}
    )

    # 启动所有任务
    task_manager.start_all()

    # 稍后可以停止或修改任务
    task_manager.stop_task("save_faiss")
    ```
    """

    def __init__(self, daemon: bool = True):
        """
        初始化任务管理器

        Args:
            daemon: 是否将线程设置为守护线程
        """
        self._tasks: Dict[str, TaskConfig] = {}
        self._task_threads: Dict[str, threading.Thread] = {}
        self._task_status: Dict[str, TaskStatus] = {}
        self._task_results: Dict[str, List[TaskResult]] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._daemon = daemon
        self._running = False
        logger.info("后台任务管理器初始化完成")

    def add_task(self, task_config: TaskConfig) -> bool:
        """添加任务"""
        with self._lock:
            if task_config.name in self._tasks:
                logger.warning(f"任务 '{task_config.name}' 已存在，将被覆盖")

            self._tasks[task_config.name] = task_config
            self._task_status[task_config.name] = TaskStatus.PENDING
            self._task_results[task_config.name] = []
            self._stop_events[task_config.name] = threading.Event()

            logger.info(f"添加任务: {task_config.name}")
            return True

    def add_interval_task(
            self,
            name: str,
            func: Callable,
            interval: timedelta,
            args: tuple = (),
            kwargs: Optional[Dict[str, Any]] = None,
            max_retries: int = 3,
            enabled: bool = True
    ) -> bool:
        """添加周期性任务"""
        task_config = TaskConfig(
            name=name,
            func=func,
            schedule_type=ScheduleType.INTERVAL,
            interval=interval,
            args=args,
            kwargs=kwargs or {},
            max_retries=max_retries,
            enabled=enabled
        )
        return self.add_task(task_config)

    def add_cron_task(
            self,
            name: str,
            func: Callable,
            cron_expression: str,
            args: tuple = (),
            kwargs: Optional[Dict[str, Any]] = None,
            max_retries: int = 3,
            enabled: bool = True
    ) -> bool:
        """添加cron表达式任务"""
        task_config = TaskConfig(
            name=name,
            func=func,
            schedule_type=ScheduleType.CRON,
            cron_expression=cron_expression,
            args=args,
            kwargs=kwargs or {},
            max_retries=max_retries,
            enabled=enabled
        )
        return self.add_task(task_config)

    def add_delayed_task(
            self,
            name: str,
            func: Callable,
            delay: timedelta,
            args: tuple = (),
            kwargs: Optional[Dict[str, Any]] = None,
            max_retries: int = 0,
            enabled: bool = True
    ) -> bool:
        """添加延迟执行任务"""
        task_config = TaskConfig(
            name=name,
            func=func,
            schedule_type=ScheduleType.DELAYED,
            delay=delay,
            args=args,
            kwargs=kwargs or {},
            max_retries=max_retries,
            enabled=enabled
        )
        return self.add_task(task_config)

    def _run_task(self, task_name: str):
        """执行任务的主循环"""
        task_config = self._tasks[task_name]
        stop_event = self._stop_events[task_name]

        logger.info(f"启动任务线程: {task_name}")

        try:
            if task_config.schedule_type == ScheduleType.DELAYED:
                # 延迟执行一次
                time.sleep(task_config.delay.total_seconds())
                if not stop_event.is_set() and task_config.enabled:
                    self._execute_task(task_name)
            else:
                # 周期性执行
                while not stop_event.is_set():
                    if task_config.enabled:
                        self._execute_task(task_name)

                    # 等待下一个执行周期
                    if task_config.schedule_type == ScheduleType.INTERVAL:
                        interval_seconds = task_config.interval.total_seconds()
                        # 分段sleep，以便能够快速响应停止信号
                        sleep_interval = min(1.0, interval_seconds / 10)
                        slept = 0
                        while slept < interval_seconds and not stop_event.is_set():
                            time.sleep(sleep_interval)
                            slept += sleep_interval
                    elif task_config.schedule_type == ScheduleType.CRON:
                        # 简化版cron调度（实际项目中可以使用python-croniter）
                        time.sleep(60)  # 每分钟检查一次
        except Exception as e:
            logger.error(f"任务线程异常: {task_name}, 错误: {e}")
        finally:
            self._task_status[task_name] = TaskStatus.STOPPED
            logger.info(f"任务线程停止: {task_name}")

    def _execute_task(self, task_name: str):
        """执行单个任务"""
        task_config = self._tasks[task_name]
        start_time = datetime.now()
        result = None
        error = None
        retry_count = 0

        with self._lock:
            self._task_status[task_name] = TaskStatus.RUNNING

        try:
            for attempt in range(task_config.max_retries + 1):
                try:
                    result = task_config.func(*task_config.args, **task_config.kwargs)
                    status = TaskStatus.COMPLETED
                    logger.debug(f"任务执行成功: {task_name}")
                    break
                except Exception as e:
                    retry_count = attempt
                    error = e

                    if attempt < task_config.max_retries:
                        logger.warning(
                            f"任务执行失败，准备重试 ({attempt + 1}/{task_config.max_retries}): "
                            f"{task_name}, 错误: {e}"
                        )
                        time.sleep(task_config.retry_delay.total_seconds())
                    else:
                        status = TaskStatus.FAILED
                        logger.error(f"任务执行失败，已达到最大重试次数: {task_name}, 错误: {e}")
        except Exception as e:
            status = TaskStatus.FAILED
            error = e
            logger.error(f"任务执行过程中发生意外错误: {task_name}, 错误: {e}")

        # 记录结果
        task_result = TaskResult(
            task_name=task_name,
            status=status,
            start_time=start_time,
            end_time=datetime.now(),
            error=error,
            result=result,
            retry_count=retry_count
        )

        with self._lock:
            self._task_status[task_name] = status
            self._task_results[task_name].append(task_result)
            # 只保留最近100次执行结果
            if len(self._task_results[task_name]) > 100:
                self._task_results[task_name] = self._task_results[task_name][-100:]

    def start_task(self, task_name: str) -> bool:
        """启动指定任务"""
        with self._lock:
            if task_name not in self._tasks:
                logger.error(f"任务不存在: {task_name}")
                return False

            if task_name in self._task_threads and self._task_threads[task_name].is_alive():
                logger.warning(f"任务已在运行: {task_name}")
                return False

            # 重置停止事件
            self._stop_events[task_name].clear()

            # 创建并启动线程
            thread = threading.Thread(
                target=self._run_task,
                args=(task_name,),
                name=f"Task-{task_name}",
                daemon=self._daemon
            )

            self._task_threads[task_name] = thread
            thread.start()
            self._task_status[task_name] = TaskStatus.RUNNING

            logger.info(f"启动任务: {task_name}")
            return True

    def start_all(self) -> bool:
        """启动所有启用的任务"""
        success = True
        with self._lock:
            for task_name, task_config in self._tasks.items():
                if task_config.enabled:
                    if not self.start_task(task_name):
                        success = False

        self._running = success
        return success

    def stop_task(self, task_name: str, timeout: float = 5.0) -> bool:
        """停止指定任务"""
        with self._lock:
            if task_name not in self._tasks:
                logger.error(f"任务不存在: {task_name}")
                return False

            # 设置停止事件
            self._stop_events[task_name].set()

            # 等待线程结束
            if task_name in self._task_threads:
                thread = self._task_threads[task_name]
                if thread.is_alive():
                    thread.join(timeout=timeout)
                    if thread.is_alive():
                        logger.warning(f"任务停止超时: {task_name}")
                        return False

            self._task_status[task_name] = TaskStatus.STOPPED
            logger.info(f"任务已停止: {task_name}")
            return True

    def stop_all(self, timeout: float = 10.0) -> bool:
        """停止所有任务"""
        success = True
        with self._lock:
            for task_name in list(self._tasks.keys()):
                if not self.stop_task(task_name, timeout=timeout):
                    success = False

        self._running = False
        return success

    def remove_task(self, task_name: str) -> bool:
        """移除任务"""
        with self._lock:
            if task_name not in self._tasks:
                return False

            # 先停止任务
            self.stop_task(task_name)

            # 清理资源
            del self._tasks[task_name]
            del self._task_status[task_name]
            del self._task_results[task_name]
            del self._stop_events[task_name]

            if task_name in self._task_threads:
                del self._task_threads[task_name]

            logger.info(f"移除任务: {task_name}")
            return True

    def update_task_config(self, task_name: str, **kwargs) -> bool:
        """更新任务配置"""
        with self._lock:
            if task_name not in self._tasks:
                logger.error(f"任务不存在: {task_name}")
                return False

            task_config = self._tasks[task_name]
            for key, value in kwargs.items():
                if hasattr(task_config, key):
                    setattr(task_config, key, value)
                else:
                    logger.warning(f"任务配置没有属性 '{key}'")

            logger.info(f"更新任务配置: {task_name}")
            return True

    def get_task_status(self, task_name: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        with self._lock:
            return self._task_status.get(task_name)

    def get_task_results(self, task_name: str, limit: int = 10) -> List[TaskResult]:
        """获取任务执行结果"""
        with self._lock:
            if task_name not in self._task_results:
                return []

            results = self._task_results[task_name]
            return results[-limit:] if limit > 0 else results

    def get_all_tasks_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有任务信息"""
        with self._lock:
            tasks_info = {}
            for name, config in self._tasks.items():
                tasks_info[name] = {
                    "status": self._task_status[name].value,
                    "enabled": config.enabled,
                    "schedule_type": config.schedule_type.value,
                    "interval": config.interval.total_seconds() if config.interval else None,
                    "last_runs": len(self._task_results.get(name, [])),
                    "is_running": self._task_threads.get(name, {}).is_alive() if name in self._task_threads else False,
                }
            return tasks_info

    def is_running(self) -> bool:
        """检查是否有任务在运行"""
        with self._lock:
            return self._running

    def __enter__(self):
        """上下文管理器入口"""
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_all()


# ============== 使用示例 ==============

# 示例1：Redis缓存中使用
class RedisCacheBackendWithTasks(RedisCacheBackend):
    """支持后台任务的Redis缓存"""

    def __init__(
            self,
            url: str,
            ttl: int = 3600,
            prefix: str = "rag:",
            embedding: Optional[embeddings] = None,
            faiss_save_path: str = "faiss_index",
            save_interval_minutes: int = 30,
            cleanup_interval_minutes: int = 60
    ):
        super().__init__(url, ttl, prefix, embedding)

        self.faiss_save_path = faiss_save_path

        # 创建后台任务管理器
        self.task_manager = BackgroundTaskManager(daemon=True)

        # 添加FAISS保存任务
        if hasattr(self, 'store') and self.store:
            self.task_manager.add_interval_task(
                name="save_faiss",
                func=self._save_faiss_task,
                interval=timedelta(minutes=save_interval_minutes),
                kwargs={"path": faiss_save_path}
            )

        # 添加Redis缓存清理任务
        self.task_manager.add_interval_task(
            name="cleanup_expired",
            func=self._cleanup_expired_task,
            interval=timedelta(minutes=cleanup_interval_minutes)
        )

        # 启动任务
        self.task_manager.start_all()

    def _save_faiss_task(self, path: str):
        """保存FAISS索引的任务函数"""
        try:
            if hasattr(self, 'store') and self.store:
                self.store.save_local(path)
                logger.info(f"FAISS索引已保存到: {path}")
                return True
        except Exception as e:
            logger.error(f"保存FAISS索引失败: {e}")
            raise

    def _cleanup_expired_task(self):
        """清理过期缓存的任务函数"""
        try:
            count = self._cleanup_expired_keys()
            if count > 0:
                logger.info(f"清理了 {count} 个过期键")
            return count
        except Exception as e:
            logger.error(f"清理过期键失败: {e}")
            raise

    def _cleanup_expired_keys(self) -> int:
        """清理过期键的实现"""
        # 这里实现具体的过期键清理逻辑
        # 使用SCAN避免阻塞
        count = 0
        cursor = 0
        pattern = f"{self.prefix}*"
        current_time = time.time()

        while True:
            cursor, keys = self.client.scan(cursor, pattern, count=100)
            if not keys:
                if cursor == 0:
                    break
                continue

            for key in keys:
                try:
                    # 检查是否过期
                    data = self.client.get(key)
                    if data:
                        rebuild_value = pickle.loads(data)
                        if rebuild_value["expiry"] < current_time:
                            # 获取原始key
                            original_key = key.decode().replace(self.prefix, "", 1)
                            self.delete(original_key)
                            count += 1
                except Exception as e:
                    logger.warning(f"清理键 {key} 时出错: {e}")

            if cursor == 0:
                break

        return count

    def close(self):
        """关闭缓存，停止后台任务"""
        # 停止所有后台任务
        self.task_manager.stop_all()

        # 保存FAISS索引
        if hasattr(self, 'store') and self.store:
            try:
                self.store.save_local(self.faiss_save_path)
            except Exception as e:
                logger.error(f"关闭时保存FAISS失败: {e}")

        # 调用父类的close（如果有）
        if hasattr(super(), 'close'):
            super().close()

        logger.info("Redis缓存已关闭")


# 示例2：内存缓存中使用
class MemoryCacheBackendWithTasks(MemoryCacheBackend):
    """支持后台任务的内存缓存"""

    def __init__(
            self,
            max_size: int = 1000,
            ttl: int = 3600,
            embedding: Optional[embeddings] = None,
            score_threshold: float = 0.95,
            cleanup_interval_minutes: int = 30
    ):
        super().__init__(max_size, ttl, embedding, score_threshold)

        # 创建后台任务管理器
        self.task_manager = BackgroundTaskManager(daemon=True)

        # 添加过期键清理任务
        self.task_manager.add_interval_task(
            name="cleanup_expired",
            func=self._cleanup_expired_task,
            interval=timedelta(minutes=cleanup_interval_minutes)
        )

        # 添加统计信息记录任务
        self.task_manager.add_interval_task(
            name="record_stats",
            func=self._record_stats_task,
            interval=timedelta(minutes=5)
        )

        # 启动任务
        self.task_manager.start_all()

    def _cleanup_expired_task(self):
        """清理过期键的任务函数"""
        try:
            count = 0
            current_time = time.time()

            with self._lock:
                # 收集过期键
                expired_keys = []
                for key, expiry in list(self._expiry.items()):
                    if current_time > expiry:
                        expired_keys.append(key)

                # 批量删除
                for key in expired_keys:
                    self._remove(key)
                    count += 1

            if count > 0:
                logger.debug(f"内存缓存清理了 {count} 个过期键")

            return count
        except Exception as e:
            logger.error(f"清理过期键失败: {e}")
            raise

    def _record_stats_task(self):
        """记录统计信息的任务函数"""
        try:
            stats = self.stats()
            # 这里可以将统计信息记录到日志、数据库或监控系统
            logger.info(f"内存缓存统计: {stats}")
            return stats
        except Exception as e:
            logger.error(f"记录统计信息失败: {e}")
            raise

    def close(self):
        """关闭缓存，停止后台任务"""
        self.task_manager.stop_all()
        logger.info("内存缓存已关闭")


# 示例3：使用上下文管理器
def example_with_context_manager():
    """使用上下文管理器管理后台任务"""
    with BackgroundTaskManager() as task_manager:
        # 添加多个任务
        task_manager.add_interval_task(
            name="task1",
            func=lambda: print("Task 1 executed"),
            interval=timedelta(seconds=10)
        )

        task_manager.add_interval_task(
            name="task2",
            func=lambda: print("Task 2 executed"),
            interval=timedelta(seconds=30)
        )

        # 在这里执行主逻辑
        # 所有任务会在with块结束时自动停止
        time.sleep(120)


# 示例4：异步任务支持（如果需要）
class AsyncTaskManager:
    """异步任务管理器（基于asyncio）"""

    def __init__(self):
        self._tasks = {}
        self._running = False

    async def add_async_task(self, name: str, coro_func, interval: float):
        """添加异步任务"""

        async def _task_wrapper():
            while self._running:
                try:
                    await coro_func()
                except Exception as e:
                    logger.error(f"异步任务 {name} 执行失败: {e}")

                await asyncio.sleep(interval)

        self._tasks[name] = asyncio.create_task(_task_wrapper())

    async def start(self):
        """启动所有异步任务"""
        self._running = True

    async def stop(self):
        """停止所有异步任务"""
        self._running = False
        for task in self._tasks.values():
            task.cancel()

        await asyncio.gather(*self._tasks.values(), return_exceptions=True)


# ============== 工厂函数模式 ==============

def create_cache_with_background_tasks(cache_type: str, **kwargs):
    """
    工厂函数：创建带后台任务的缓存

    Args:
        cache_type: "redis" 或 "memory"
        **kwargs: 传递给缓存的参数

    Returns:
        配置好后台任务的缓存实例
    """
    if cache_type == "redis":
        return RedisCacheBackendWithTasks(**kwargs)
    elif cache_type == "memory":
        return MemoryCacheBackendWithTasks(**kwargs)
    else:
        raise ValueError(f"不支持的缓存类型: {cache_type}")