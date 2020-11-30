module.exports = {
	plugins: ['commitlint-plugin-jira-rules'],
	extends: ['jira'],
	rules: {
		"header-max-length": [2, "always", 88],
		"signed-off-by": [2, "always", "Signed-off-by"],
		"jira-task-id-max-length": [0, "always", 10],
		"jira-task-id-project-key": [2, "always", ["FEAT", "DOCS", "FIX", "REFACTOR", "TEST"]],
	}
}
