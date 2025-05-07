class Host:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patched_vulnerabilities = []