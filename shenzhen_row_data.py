import time
from pathlib import Path

import pandas as pd
import requests


def _build_output_csv(start_date: str, end_date: str) -> Path:
    """根据日期范围生成输出 CSV 路径，位于项目 data 目录。"""
    project_root = Path(__file__).resolve().parent.parent.parent
    csv_dir = project_root / "data" / "raw"
    csv_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"bike_orders_{start_date}_{end_date}.csv"
        if start_date != end_date
        else f"bike_orders_{start_date}.csv"
    ) # 如果你要修改导出的csv的文件吗，请修改这里。
    return csv_dir / filename


# 主函数
if __name__ == "__main__":
    # 环境变量和初始化
    app_key = "3f95d4977191458aa3492a689-----"  # TODO: 替换为你从深圳开放数据平台申请的 app_key
    page_num = 1
    rows = 4000
    # 日期范围 可以选择你要爬取的范围
    startDate = "20210501"  # TODO: 替换为你要爬取的开始日期，格式 YYYYMMDD
    endDate = "20210503"  # TODO: 替换为你要爬取的结束日期，格式 YYYYMMDD
    url = "https://opendata.sz.gov.cn/api/29200_00403627/1/service.xhtml"
    # 请求头 不加请求会被拒绝
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }

    # 输出 CSV
    output_csv = _build_output_csv(startDate, endDate)
    print(f"数据将保存到: {output_csv}")

    # 数据请求和处理循环
    while True:
        params = {
            "appKey": app_key,
            "page": page_num,
            "rows": rows,
            "startDate": startDate,
            "endDate": endDate,
        }
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"请求错误，状态码：{response.status_code}")
            break

        items = response.json().get("data", [])
        if not items:
            print("没有更多数据或数据为空，结束。")
            break

        # 将本页数据追加写入 CSV
        df = pd.DataFrame(items)
        write_header = (page_num == 1) and (not output_csv.exists())
        df.to_csv(
            output_csv, mode="a", index=False, encoding="utf-8-sig", header=write_header
        )
        print(f"已写入第 {page_num} 页，共 {len(df)} 条。")

        # 判断是否继续
        if len(items) < rows:
            print("最后一页已写完。")
            break
        else:
            page_num += 1
            # 轻微休眠，避免请求过快
            time.sleep(0.5)