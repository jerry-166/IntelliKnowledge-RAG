from typing import Optional

from fastapi import FastAPI, Path, Query, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

"""
1.路由优先级原则
具体路由优先：固定路径必须置于动态路径之前
2. 异步处理原则
异步函数：路由函数建议使用 async def
异步中间件：中间件需配合 await call_next(request)
性能优化：合理使用异步提高并发处理能力
3. 依赖注入原则
可复用组件：将公共逻辑封装为依赖函数
生命周期管理：理解依赖的执行时机和作用域
参数传递：依赖可接受参数并返回数据供路由使用
"""


app = FastAPI()  # 创建fastapi实例


# 中间件（过滤器）
@app.middleware(middleware_type="http")
async def middleware1(request, call_next):
    print("Middleware1 进入")
    response = await call_next(request)  # 异步逻辑需await
    print("Middleware1 退出")
    return response


@app.middleware(middleware_type="http")
async def middleware2(request, call_next):
    print("Middleware2 进入")
    response = await call_next(request)  # 异步逻辑需await
    print("Middleware2 退出")
    return response


@app.get("/")
def hello():
    return {"message": "hello world"}


# 路径参数
@app.get("/hello/{name}")
async def hello(name: str = Path(..., title="a", description="姓名", min_length=2, max_length=10)):
    return {"name": name, "desc": f"我是{name}"}


# 查询参数
@app.get("/book")
async def get_book(
        category: str = Query("Python开发", min_length=5, max_length=50, description="分类"),
        price: int = Query(..., ge=50, le=100, description="价格")
):
    return {
        "category": category,
        "price": price
    }


class Book(BaseModel):
    name: str = Field(..., min_length=2, max_length=20)  # ... 表示必填，换成具体值则表示默认值
    author: str = Field(..., min_length=2, max_length=10)  # ...表示必填，换成具体值则表示有默认值
    price: Optional[int] = Field(None, gt=0)  # 选填，带验证


# 请求体参数
@app.post("/book/add")
async def add_book(book: Book):
    return {
        "msg": f"添加{book.author}图书{book.name}成功，价格为：{book.price}"
    }


# html响应
@app.get("/html", response_class=HTMLResponse)
async def get_html():
    return "<h1>一级标题</h1>"


# 文件、视频等响应
@app.get("/file")
async def get_file():
    path = r"C:\Users\ASUS\Desktop\IntelliKnowledge-RAG\python_backend\fastApiProject\美女.png"
    return FileResponse(path=path, filename="美女")


class User(BaseModel):
    id: int
    name: str
    gender: str


# 自定义返回格式
@app.get("/user/{id}", response_model=User)
async def get_user(id: int = Path(...)):
    # return {
    #     "id": id,
    #     "name": "zhangsan1",
    #     "gender": "male1"
    # }
    return User(id=id, name="zhangsan2", gender="male2")


# 依赖注入
async def common_parameters(
        skip: int = Query(..., ge=0),
        size: int = Query(..., le=100)
):
    return {
        "skip": skip,
        "size": size
    }


@app.get("/news/list")
async def get_news(commons=Depends(common_parameters)):
    return commons


@app.get("/book/list")
async def get_news():
    return {
        "msg": "aaa"
    }


# 错误处理
@app.get("/news/{id}")
async def get_new(id: int):
    id_list = [1, 2, 3]
    if id not in id_list:
        raise HTTPException(status_code=404, detail="未知id")
    return {
        "id": id,
    }
