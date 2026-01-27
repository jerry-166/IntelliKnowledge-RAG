from pydantic import BaseModel, Field
from typing import Optional, Any


class ErrorResponse(BaseModel):
    """统一的错误响应模型"""
    success: bool = Field(default=False, description="请求是否成功")
    error_code: str = Field(..., description="错误码")
    message: str = Field(..., description="错误信息")
    details: Optional[Any] = Field(None, description="错误详情")
    solution: Optional[str] = Field(None, description="解决方案建议")


class SuccessResponse(BaseModel):
    """统一的成功响应模型"""
    success: bool = Field(default=True, description="请求是否成功")
    message: Optional[str] = Field(None, description="成功信息")
    data: Optional[Any] = Field(None, description="响应数据")
