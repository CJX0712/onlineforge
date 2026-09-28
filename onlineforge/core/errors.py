"""错误码与异常。E1xx 输入 / E2xx 训练 / E3xx 预测 / E4xx 漂移 / E5xx 配置。"""


class ForgeError(Exception):
    """OnlineForge 基类异常。"""

    code = "E000"

    def __init__(self, message: str, code: str | None = None):
        self.code = code or self.code
        super().__init__(f"[{self.code}] {message}")


class BadInputError(ForgeError):
    code = "E100"


class FitError(ForgeError):
    code = "E200"


class PredictError(ForgeError):
    code = "E300"


class DriftError(ForgeError):
    code = "E400"


class ConfigError(ForgeError):
    code = "E500"


E100_BAD_INPUT = "E100"
E200_FIT = "E200"
E300_PREDICT = "E300"
E400_DRIFT = "E400"
E500_CONFIG = "E500"
