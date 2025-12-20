from datetime import datetime
from typing import Literal, Any, Optional, Dict, List, Union

from qdrant_client.http.models import (
    FieldCondition, Filter, MatchAny, Range, MatchExcept, MatchValue
)


class SearchKwargsUtil:
    @staticmethod
    def build_search_kwargs(
            vector_store: Literal["milvus", "faiss", "qdrant", "chroma"],
            search_kwargs: Optional[Dict[str, Any]] = None,
            filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        高级过滤检索函数：支持等值/时间范围/不等于/多值匹配/空值/数值范围/逻辑组合
        统一条件格式说明（支持的操作符）：
        - 等值：{"key": "value"} 或 {"key": {"$eq": "value"}}
        - 不等于：{"key": {"$ne": "value"}}
        - 多值匹配：{"key": {"$in": ["value1", "value2"]}}
        - 不包含：{"key": {"$nin": ["value1", "value2"]}}
        - 空值/非空：{"key": {"$exists": False/True}}
        - 范围（时间/数值）：{"key": {"$gt": 100, "$lt": 1000, "$gte": "2025-12-16", "$lte": "2025-12-18"}}
        - 逻辑组合：{"$and": [cond1, cond2]}, {"$or": [cond1, cond2]}, {"$not": cond}
        """
        search_kwargs = search_kwargs or {"k": 5}
        filter_conditions = filter_conditions or {}

        # ------------------------------
        # 辅助函数：判断是否为日期字符串
        # ------------------------------
        def is_date_string(value: Any) -> bool:
            """判断字符串是否为日期格式"""
            if not isinstance(value, str):
                return False
            # 尝试常见日期格式
            date_formats = [
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y/%m/%d",
                "%Y/%m/%d %H:%M:%S"
            ]
            for fmt in date_formats:
                try:
                    datetime.strptime(value, fmt)
                    return True
                except ValueError:
                    continue
            return False

        # ------------------------------
        # 辅助函数：日期字符串转时间戳（浮点数）
        # ------------------------------
        def date_to_timestamp(value: Any) -> Union[float, str]:
            """将日期字符串或datetime对象转换为时间戳"""
            if isinstance(value, datetime):
                return value.timestamp()
            elif is_date_string(value):
                # 尝试常见日期格式
                date_formats = [
                    "%Y-%m-%d",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y/%m/%d",
                    "%Y/%m/%d %H:%M:%S"
                ]
                for fmt in date_formats:
                    try:
                        dt = datetime.strptime(value, fmt)
                        return dt.timestamp()
                    except ValueError:
                        continue
                return value  # 如果无法解析，返回原值
            return value  # 非日期值直接返回

        # ------------------------------
        # 核心工具函数：根据向量数据库类型格式化值
        # ------------------------------
        def format_time(value: Any, vector_db: str = "chroma") -> Any:
            """
            将时间格式根据向量数据库类型进行转换
            - Chroma/FAISS: 日期字符串或ISO字符串
            - Qdrant: 日期转为时间戳（浮点数）
            - Milvus: 日期转为字符串
            """
            if isinstance(value, datetime):
                if vector_db == "qdrant":
                    return value.timestamp()  # Qdrant需要时间戳
                else:
                    return value.isoformat()  # 其他数据库用ISO字符串
            elif isinstance(value, str) and is_date_string(value):
                if vector_db == "qdrant":
                    return date_to_timestamp(value)  # Qdrant需要时间戳
                else:
                    # 其他数据库保持原日期字符串或转为ISO
                    return value
            elif isinstance(value, (int, float)):
                return value
            else:
                return str(value)

        # ------------------------------
        # 1. Chroma/FAISS 处理
        # ------------------------------
        if vector_store in ["chroma", "faiss"]:
            def format_conditions(cond: Dict[str, Any]) -> Dict[str, Any]:
                formatted = {}
                for key, value in cond.items():
                    if key in ["$and", "$or", "$not"]:
                        if key == "$not":
                            formatted[key] = format_conditions(value) if isinstance(value, dict) else value
                        else:
                            formatted[key] = [format_conditions(v) for v in value]
                    elif isinstance(value, dict) and any(k in value for k in ["$gt", "$lt", "$gte", "$lte"]):
                        formatted_val = {}
                        for op, v in value.items():
                            # Chroma/FAISS用ISO字符串
                            formatted_val[op] = format_time(v, "chroma")
                        formatted[key] = formatted_val
                    else:
                        formatted[key] = format_time(value, "chroma")
                return formatted

            final_kwargs = format_conditions(filter_conditions)
            search_kwargs["filter"] = final_kwargs

        # ------------------------------
        # 2. Qdrant 处理（修复核心：Range需要数值类型）
        # ------------------------------
        elif vector_store == "qdrant":
            def build_qdrant_filter(cond: Dict[str, Any]) -> Union[Filter, FieldCondition]:
                """递归构建 Qdrant Filter/FieldCondition 对象"""
                # 处理逻辑运算符（$and/$or/$not）
                if "$and" in cond:
                    sub_filters = [build_qdrant_filter(c) for c in cond["$and"]]
                    return Filter(must=sub_filters)
                elif "$or" in cond:
                    sub_filters = [build_qdrant_filter(c) for c in cond["$or"]]
                    return Filter(should=sub_filters)
                elif "$not" in cond:
                    sub_filter = build_qdrant_filter(cond["$not"])
                    return Filter(must_not=[sub_filter])

                # 处理普通字段条件
                field_conditions = []
                for key, val in cond.items():
                    if isinstance(val, dict):
                        # 带操作符的条件
                        if "$eq" in val:
                            formatted_val = format_time(val["$eq"], "qdrant")
                            field_conditions.append(
                                FieldCondition(key=key, match=MatchValue(value=formatted_val))
                            )
                        elif "$ne" in val:
                            formatted_val = format_time(val["$ne"], "qdrant")
                            field_conditions.append(
                                FieldCondition(key=key, match=MatchExcept(except_=[formatted_val]))
                            )
                        elif "$in" in val:
                            formatted_vals = [format_time(v, "qdrant") for v in val["$in"]]
                            field_conditions.append(FieldCondition(key=key, match=MatchAny(any=formatted_vals)))
                        elif "$nin" in val:
                            formatted_vals = [format_time(v, "qdrant") for v in val["$nin"]]
                            field_conditions.append(FieldCondition(key=key, match=MatchExcept(except_=formatted_vals)))
                        elif "$exists" in val:
                            field_conditions.append(FieldCondition(key=key, is_empty=not val["$exists"]))
                        elif any(op in val for op in ["$gt", "$gte", "$lt", "$lte"]):
                            # 范围条件：Qdrant的Range需要数值类型，所以日期要转为时间戳
                            range_params = {}
                            for op, v in val.items():
                                if op in ["$gt", "$gte", "$lt", "$lte"]:
                                    # 转换日期为时间戳
                                    converted_value = format_time(v, "qdrant")
                                    # 确保值是数值类型
                                    if isinstance(converted_value, (int, float)):
                                        # 去掉$符号作为Range参数名
                                        range_key = op.replace("$", "")
                                        range_params[range_key] = float(converted_value)

                            # 验证range_params（避免空字典）
                            if not range_params:
                                continue

                            # 创建Range过滤器
                            field_conditions.append(
                                FieldCondition(key=key, range=Range(**range_params))
                            )
                    else:
                        # 简化等值条件
                        formatted_val = format_time(val, "qdrant")
                        field_conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=formatted_val))
                        )

                # 组合多个字段条件
                if len(field_conditions) == 1:
                    return field_conditions[0]
                else:
                    return Filter(must=field_conditions)

            qdrant_filter = build_qdrant_filter(filter_conditions)
            search_kwargs["filter"] = qdrant_filter

        # ------------------------------
        # 3. Milvus 处理
        # ------------------------------
        elif vector_store == "milvus":
            def build_milvus_expr(cond: Dict[str, Any]) -> str:
                expr_parts = []
                for key, val in cond.items():
                    if key == "$and":
                        sub_exprs = [build_milvus_expr(c) for c in val]
                        expr_parts.append(f"({' AND '.join(sub_exprs)})")
                    elif key == "$or":
                        sub_exprs = [build_milvus_expr(c) for c in val]
                        expr_parts.append(f"({' OR '.join(sub_exprs)})")
                    elif key == "$not":
                        sub_expr = build_milvus_expr(val)
                        expr_parts.append(f"(NOT {sub_expr})")
                    elif isinstance(val, dict):
                        if "$eq" in val:
                            v = format_time(val["$eq"], "milvus")
                            expr_parts.append(f"{key} = '{v}'" if isinstance(v, str) else f"{key} = {v}")
                        elif "$ne" in val:
                            v = format_time(val["$ne"], "milvus")
                            expr_parts.append(f"{key} != '{v}'" if isinstance(v, str) else f"{key} != {v}")
                        elif "$in" in val:
                            vals = [
                                f"'{format_time(v, 'milvus')}'" if isinstance(v, str) else str(format_time(v, 'milvus'))
                                for v in val["$in"]]
                            expr_parts.append(f"{key} IN ({', '.join(vals)})")
                        elif "$nin" in val:
                            vals = [
                                f"'{format_time(v, 'milvus')}'" if isinstance(v, str) else str(format_time(v, 'milvus'))
                                for v in val["$nin"]]
                            expr_parts.append(f"{key} NOT IN ({', '.join(vals)})")
                        elif "$exists" in val:
                            expr_parts.append(f"{key} IS NOT NULL" if val["$exists"] else f"{key} IS NULL")
                        elif any(op in val for op in ["$gt", "$gte", "$lt", "$lte"]):
                            range_parts = []
                            for op, v in val.items():
                                if op in ["$gt", "$gte", "$lt", "$lte"]:
                                    formatted_v = format_time(v, "milvus")
                                    op_symbol = op.replace("$", "")
                                    if isinstance(formatted_v, str):
                                        range_parts.append(f"{key} {op_symbol} '{formatted_v}'")
                                    else:
                                        range_parts.append(f"{key} {op_symbol} {formatted_v}")
                            expr_parts.append(f"({' AND '.join(range_parts)})")
                    else:
                        v = format_time(val, "milvus")
                        expr_parts.append(f"{key} = '{v}'" if isinstance(v, str) else f"{key} = {v}")
                return " AND ".join(expr_parts)

            milvus_expr = build_milvus_expr(filter_conditions)
            search_kwargs["expr"] = milvus_expr if milvus_expr else ""

        else:
            raise ValueError(f"不支持的向量库类型：{vector_store}")

        return search_kwargs


if __name__ == "__main__":
    # 示例 1：Chroma 多条件过滤
    print("=== 示例1：Chroma 多条件过滤 ===")
    filter_cond1 = {
        "$and": [
            {"create_time": {"$gte": "2025-12-16"}},
            {"source": {"$in": ["PDF", "MD"]}},
            {"texts_length": {"$gt": 200, "$lt": 2000}}
        ]
    }
    search_kwargs1 = SearchKwargsUtil.build_search_kwargs(
        vector_store="chroma",
        filter_conditions=filter_cond1,
        search_kwargs={"k": 10}
    )
    print(search_kwargs1)

    # 示例 2：Milvus 排除条件过滤
    print("\n=== 示例2：Milvus 排除条件过滤 ===")
    filter_cond2 = {
        "$and": [
            {"type": {"$ne": "图片"}},
            {"source": {"$exists": True}},
            {"source": {"$nin": ["TXT"]}}
        ]
    }
    search_kwargs2 = SearchKwargsUtil.build_search_kwargs(
        vector_store="milvus",
        filter_conditions=filter_cond2
    )
    print(search_kwargs2)

    # 示例 3：Qdrant 复杂逻辑过滤（修复后无报错）
    print("\n=== 示例3：Qdrant 复杂逻辑过滤 ===")
    filter_cond3 = {
        "$or": [
            {"$and": [
                {"source": "PDF"},
                {"texts_length": {"$gt": 500}}  # 数值范围
            ]},
            {"$and": [
                {"source": "MD"},
                {"create_time": {"$lte": "2025-12-10"}}  # 时间范围
            ]}
        ]
    }
    search_kwargs3 = SearchKwargsUtil.build_search_kwargs(
        vector_store="qdrant",
        filter_conditions=filter_cond3
    )
    print(search_kwargs3)

    # 示例 4：Qdrant 时间范围+数值范围混合过滤
    print("\n=== 示例4：Qdrant 混合范围过滤 ===")
    filter_cond4 = {
        "$and": [
            {"update_time": {"$gt": "2025-12-01", "$lt": datetime(2025, 12, 20)}},
            {"score": {"$gte": 80.5, "$lte": 100}}
        ]
    }
    search_kwargs4 = SearchKwargsUtil.build_search_kwargs(
        vector_store="qdrant",
        filter_conditions=filter_cond4
    )
    print(search_kwargs4)