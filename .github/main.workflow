workflow "Code Style" {
  on = "push"
  resolves = ["lint-action"]
}

action "lint-action" {
  uses = "CyberZHG/github-action-python-lint@master"
  args = "--max-line-length=120 mxnet_octave_conv gluon_octave_conv tests"
}
