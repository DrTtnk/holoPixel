import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 120000,
  retries: 0,
  use: {
    screenshot: 'on',
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'electron',
      testMatch: /.*\.e2e\.ts$/,
    },
  ],
});
