"""Network routing and retry tests without installing or downloading packages."""

import unittest

import make_failure_pilot_notebook as notebook


class SetupNetworkTests(unittest.TestCase):
    def scope(self):
        namespace = {"VALIDATE_ONLY": True}
        exec(notebook.HELPERS, namespace)
        return namespace

    def test_domestic_index_bypasses_proxies_without_mutating_turbo_environment(self):
        base = {key: "private-proxy-fixture" for key in (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
            "http_proxy", "https_proxy", "all_proxy", "no_proxy")}
        base["UV_CACHE_DIR"] = "retained-cache"
        original = dict(base)
        direct = self.scope()["package_environment"](base)
        self.assertEqual(base, original)
        self.assertFalse(any(key.lower().endswith("proxy") for key in direct))
        self.assertEqual(direct["UV_CACHE_DIR"], "retained-cache")
        self.assertEqual(direct["UV_HTTP_TIMEOUT"], "600")
        self.assertEqual(direct["UV_CONCURRENT_DOWNLOADS"], "2")

    def test_install_falls_back_to_alternate_index_and_keeps_package_pins(self):
        ns = self.scope()
        calls = []
        def logged(command, name, env):
            calls.append((command, env))
            if len(calls) == 1:
                raise RuntimeError("synthetic timeout")
        ns["logged"] = logged
        ns["install_packages"](["python", "-m", "uv"], "venv/python", ["numpy==1.26.4"],
                               {"HTTPS_PROXY": "private", "UV_CACHE_DIR": "cached"}, ["--no-deps"])
        self.assertEqual(len(calls), 2)
        self.assertIn("https://mirrors.aliyun.com/pypi/simple", calls[1][0])
        self.assertIn("numpy==1.26.4", calls[1][0])
        self.assertIn("--no-deps", calls[1][0])
        self.assertNotIn("HTTPS_PROXY", calls[1][1])
        self.assertEqual(calls[1][1]["UV_CACHE_DIR"], "cached")

    def test_both_index_failures_stop_instead_of_reporting_success(self):
        ns = self.scope()
        def failed(*args, **kwargs):
            raise RuntimeError("synthetic timeout")
        ns["logged"] = failed
        with self.assertRaisesRegex(RuntimeError, "both indexes"):
            ns["install_packages"](["uv"], "python", ["pinned-package==1"], {})


if __name__ == "__main__":
    unittest.main()
