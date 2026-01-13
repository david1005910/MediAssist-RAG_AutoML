/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Frosted Metal Palette
        metal: {
          // Light Metal
          light: '#E3E6EB',
          'light-mid': '#C9CED6',
          'light-dark': '#B6BCC7',
          // Dark Metal / Titanium
          dark: '#4A4F59',
          'dark-mid': '#373B44',
          'dark-deep': '#2C3036',
          // Blue Anodized
          'blue': '#3C6E96',
          'blue-mid': '#355E82',
          'blue-deep': '#2C4F6A',
        },
        // Cool Accents
        accent: {
          cyan: '#4FC3F7',
          'cyan-light': '#81D4FA',
          'cyan-bright': '#29B6F6',
          green: '#26C281',
          'green-light': '#2ECC71',
        },
        // Text Colors
        'metal-text': {
          light: '#F0F0F0',
          mid: '#D1D5DB',
          muted: '#A8B0BA',
          dark: '#4A4F59',
        },
        // Legacy support
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        medical: {
          blue: '#0077b6',
          green: '#2a9d8f',
          red: '#e63946',
        },
      },
      backgroundImage: {
        // Metallic Gradients
        'metal-silver': 'linear-gradient(180deg, #E3E6EB 0%, #C9CED6 50%, #B6BCC7 100%)',
        'metal-titanium': 'linear-gradient(180deg, #4A4F59 0%, #373B44 50%, #2C3036 100%)',
        'metal-blue': 'linear-gradient(180deg, #3C6E96 0%, #355E82 50%, #2C4F6A 100%)',
        'metal-dark': 'linear-gradient(180deg, #2C3036 0%, #23272B 50%, #1A1D21 100%)',
        'metal-surface': 'linear-gradient(180deg, #3A3F47 0%, #2F343B 50%, #252930 100%)',
      },
      boxShadow: {
        'metal': '0px 4px 16px rgba(0,0,0,0.20)',
        'metal-sm': '0px 2px 8px rgba(0,0,0,0.15)',
        'metal-lg': '0px 8px 24px rgba(0,0,0,0.25)',
        'metal-inset': 'inset 0px 1px 0px rgba(255,255,255,0.15), inset 0px -1px 0px rgba(0,0,0,0.3)',
      },
      borderRadius: {
        'metal': '6px',
        'metal-sm': '4px',
        'metal-lg': '8px',
      },
    },
  },
  plugins: [],
}
