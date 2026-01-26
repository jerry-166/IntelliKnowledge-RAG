from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Path, Depends, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import DateTime, func, String, Float, Integer, select, Index, update, and_
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# 1.安装库 sqlalchemy[asyncio]、aiomysql
# 2.异步引擎
ASYNC_ENGINE_URL = "mysql+aiomysql://root:1234@localhost:3306/fastapi_test?charset=utf8"
async_engine = create_async_engine(
    url=ASYNC_ENGINE_URL,
    echo=True,  # 打印sql信息
    pool_size=5,  # 线程池开放数
    max_overflow=10,  # 超出pool_size后的最大连接数
)


# 3.建库---自己建
# 4.建表：基类 + 表对应的模型类
class Base(DeclarativeBase):
    create_time: Mapped[datetime] = mapped_column(DateTime, insert_default=func.now(), default=func.now,
                                                  comment="创建时间")
    update_time: Mapped[datetime] = mapped_column(DateTime, insert_default=func.now(), default=func.now,
                                                  onupdate=func.now(), comment="更新时间")


class User(Base):
    __tablename__ = "user"
    # 注意如果是单个，要加逗号comma，不然无法将一个(Index)识别成元组
    __table_args__ = (
        Index("idx_score", "score"),  # 索引，查询勤、排序字段 --- 建议加
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(255), nullable=True, comment="用户名")
    password: Mapped[str] = mapped_column(String(255), comment="密码")
    score: Mapped[float] = mapped_column(Float(2), comment="分数")

    def __repr__(self):
        return f"<User(id:{self.id}, username:{self.username}, password:{self.password}, score:{self.score})>"


# 4. Pydantic 模型（仅用于接口参数校验/序列化）
# 新增：创建用户的请求体模型（无需id，自增）
class UserCreate(BaseModel):
    username: str | None = None
    password: str
    score: float


# 新增：修改用户的请求体模型（需id）
class UserUpdate(BaseModel):
    id: int
    username: str | None = None
    password: str | None = None
    score: float | None = None


# 4.1 在调用Fastapi启动时，运行建表语句
async def create_tables():
    # 获取异步引擎，创建事务 --- 建表
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 加载模型前
    await create_tables()
    # 执行
    yield
    # 执行后，清理


# 会话工厂
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,  # 会话类型，默认值
    expire_on_commit=False,  # 提交过期吗？ False - 不会重新提交，快
)


# 依赖项，通过工厂获取db
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session  # 返回session对象
            await session.commit()  # 用完之后，会话提交
        except Exception:
            await session.rollback()  # 报错，则回滚
            raise
        finally:
            await session.close()  # 最后关闭会话


app = FastAPI(lifespan=lifespan)

# 添加处理跨域的中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源访问，实际上线需自定义origins[]，添加所允许的网址
    allow_credentials=True,  # 允许携带cookie
    allow_headers=["*"],  # 允许请求头
    allow_methods=["*"],  # 允许任何方法，post、get、put、delete
)
router = APIRouter(prefix="/user", tags=["user"])


@router.get("/avg_score")
async def get_user_avg_score(
        db: AsyncSession = Depends(get_db)
):
    # 聚合查询
    stmt = select(func.avg(User.score).label("avg_score"))
    result = await db.execute(stmt)
    avg_score = result.scalar_one_or_none()
    return avg_score


@router.get("/list")
async def get_users(
        db: AsyncSession = Depends(get_db)
):
    # 分页排序 + 索引
    stmt2 = select(User).order_by(User.score.desc(), User.create_time).offset(0).limit(2)
    result = await db.execute(stmt2)
    users = result.scalars().all()
    if users is None:
        raise HTTPException(404, "没有用户")
    return users


@router.get("/{id}")
async def get_user(
        id: int = Path(..., description="用户id"),
        db: AsyncSession = Depends(get_db)
):
    # 指定查询
    stmt = select(User).where(and_(User.id == id, User.score > 10))
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    # user = await db.get(User, 1)
    return user


@router.post("/add")
async def add_user(
        user: UserCreate,
        db: AsyncSession = Depends(get_db),
):
    # 增加用户
    user = User(**user.__dict__)
    db.add(user)
    await db.commit()
    return {"status": "true"}


@router.delete("/delete")
async def delete_user(
        id: int,
        db: AsyncSession = Depends(get_db),
):
    # 先查
    stmt = select(User).where(User.id == id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(404, "未知id")
    # 后删
    await db.delete(user)
    await db.commit()
    return user


@router.put("/modify")
async def modify_user(
        user: UserUpdate,
        db: AsyncSession = Depends(get_db)
):
    id = user.id
    # 使用pydantic的model_dump获得字典
    update_data = user.model_dump(exclude_unset=True, exclude={"id"})
    # 法一：
    if id % 2 == 0:
        # 先查
        t_user = await db.get(User, id)
        # 后改
        for k, v in update_data.items():
            setattr(t_user, k, v)
        await db.commit()
    # 法二：
    else:
        # 使用update()
        stmt = update(User).where(User.id == id).values(**update_data)
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount > 0

    return {"msg": "修改成功"}


# 修复：将include_router移到所有路由函数定义之后
app.include_router(router=router)  # 加入router
