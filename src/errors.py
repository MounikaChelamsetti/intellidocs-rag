"""Exception hierarchy. Every error carries a message that is safe to show to end users
(no stack traces, no secrets)."""


class IntelliDocsError(Exception):
    """Base class for all expected, user-presentable errors."""


class ConfigError(IntelliDocsError):
    """Missing or invalid configuration (e.g. no API key)."""


class PDFValidationError(IntelliDocsError):
    """The uploaded file is not an acceptable PDF (type, size, page count)."""


class PDFProcessingError(IntelliDocsError):
    """The PDF is acceptable on the surface but could not be read."""


class GeminiAuthError(IntelliDocsError):
    """The Gemini API rejected the credentials."""


class GeminiRateLimitError(IntelliDocsError):
    """Rate limit or quota exhausted (after retries)."""


class GeminiServiceError(IntelliDocsError):
    """Any other Gemini API failure (network, 5xx, blocked/empty response)."""
