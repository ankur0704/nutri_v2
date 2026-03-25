// Central API configuration
// Set VITE_API_URL in your .env file to override for staging/production
const API_BASE = import.meta.env.VITE_API_URL || '';

export default API_BASE;
