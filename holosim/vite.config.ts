import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import electron from 'vite-plugin-electron';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig(() => {
  return {
    plugins: [
      react(),
      tailwindcss(),
      electron([
        {
          entry: 'electron/main.ts',
          vite: {
            build: {
              outDir: 'dist-electron',
              rollupOptions: {
                external: ['electron', /\.node$/],
                output: {
                  format: 'cjs',
                },
              },
            },
          },
        },
        {
          entry: 'electron/preload.ts',
          onstart(args) {
            args.reload();
          },
          vite: {
            build: {
              outDir: 'dist-electron',
              rollupOptions: {
                output: {
                  format: 'cjs',
                },
              },
            },
          },
        },
      ]),
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      hmr: process.env.DISABLE_HMR !== 'true',
    },
  };
});
