# shellcheck shell=bash
# 国内行情源直连白名单 —— 被 weekly_data.sh / morning_brief.sh / daily_run.sh source。
#
# 背景（2026-08-05 定位）：本机 macOS 系统代理常开（127.0.0.1:7897），
# requests 在 macOS 上会读系统代理配置，于是东财等国内接口全部绕道代理出国再回来，
# 大面积 ProxyError。8/2 周维护实测 em_fundflow 2904/2949 失败(98.4%)，
# 加 NO_PROXY 直连后同批抽样 8/8 全成。
# 注：这不是"东财封 IP"（早前的错误结论），是本机代理在中间掐。
#
# 只列国内源；GitHub / qlib release 仍走代理（bootstrap 需要）。
# requests/urllib 按域名后缀匹配，写主域即可覆盖所有子域（push2his.eastmoney.com 等）。

_NO_PROXY_DOMAINS="\
eastmoney.com,\
sina.com.cn,sinajs.cn,\
10jqka.com.cn,\
swsindex.com,swhyresearch.com,\
cninfo.com.cn,sse.com.cn,szse.cn,csindex.com.cn,\
baidu.com,\
gtimg.cn,qq.com,\
xueqiu.com,legulegu.com,\
tushare.pro,akfamily.xyz,\
localhost,127.0.0.1"

export NO_PROXY="$_NO_PROXY_DOMAINS"
export no_proxy="$_NO_PROXY_DOMAINS"   # requests 认小写，curl 认小写，两个都设
