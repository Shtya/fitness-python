const path = require('path');

const isWin = process.platform === 'win32';
const python = isWin
	? path.join(__dirname, '.venv', 'Scripts', 'python.exe')
	: path.join(__dirname, '.venv', 'bin', 'python');

module.exports = {
	apps: [
		{
			name: 'ai-body-scan',
			cwd: __dirname,
			script: python,
			args: '-m uvicorn main:app --host 127.0.0.1 --port 8010',
			interpreter: 'none',
			instances: 1,
			exec_mode: 'fork',
			autorestart: true,
			watch: false,
			max_memory_restart: '1500M',
			env: {
				PYTHONUNBUFFERED: '1',
			},
		},
	],
};
