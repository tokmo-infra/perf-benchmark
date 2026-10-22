class PipelineError(RuntimeError):
    pass


class PluginNotFound(PipelineError):
    pass


class HookNotFound(PipelineError):
    pass


class ResultArchiveError(PipelineError):
    """Committed benchmark facts survive a failed immutable archive delivery."""
