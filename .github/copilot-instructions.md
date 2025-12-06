# GitHub Copilot Instructions

You are an expert developer working on the "HackNation2025-Hackaton" project.

## Project Structure
- `/dashboard`: A React Router v7 (Remix-style) frontend application.
- `/data`: Contains raw data files (CSV, XML) and documentation.
- `/data-processing`: Contains Python scripts and Jupyter notebooks for data analysis.

## Tech Stack
- **Frontend**: React, React Router v7, TypeScript, Vite.
- **Data Science**: Python, Pandas, Jupyter.

## Coding Guidelines

### TypeScript / React (Dashboard)
- Use **functional components** and **hooks**.
- Use **TypeScript** for all new files (`.ts`, `.tsx`).
- Prefer **interfaces** over types for object definitions.
- Use `loader` and `action` functions for data fetching and mutations in React Router.
- Ensure accessibility (a11y) standards are met.

### Python (Data Processing)
- Use **Pandas** for data manipulation.
- Document steps in Jupyter notebooks clearly.
- Handle data paths relative to the project root or script location appropriately.

## General
- Keep answers concise and relevant.
- When generating code, ensure it fits the existing project structure.
- If modifying configuration files (like `vite.config.ts` or `tsconfig.json`), explain the changes.



# Research Summary: PKO BP Brandbook & React Dashboard Implementation
PKO BP Brand Identity Overview
Historical Context & Current Design System

PKO BP underwent a significant rebranding (around 2010) by White Cat Studio, moving away from its PRL-era perception. The key elements include:

Primary Colors:

"Polish" dark navy/red gradient

Black as refined accent

Gradient-based color system

Typography:

Custom corporate font (designed specifically for PKO)

Emphasis on modern, clean hierarchy

The PKO signet plays a central role in the identity system

Design Philosophy:

Mathematical precision (grid-based patterns with sinusoidal lines)

Trust and modernity

Balance between aesthetics and mathematical order

Key Brandbook Characteristics for Digital Implementation
Color Palette (for React/CSS):

css
--pko-navy: #1a2f3a;        /* Primary dark */
--pko-red: #d93026;         /* Brand red */
--pko-gold: #c9a961;        /* Accent (Private Banking) */
--pko-black: #1a1a1a;       /* Deep black */
--pko-white: #ffffff;
--pko-gray-light: #f5f5f5;
--pko-gray-medium: #d0d0d0;
Typography System:

Use the custom PKO font if available (enterprise license)

Fallback: System fonts like -apple-system, BlinkMacSystemFont, 'Segoe UI'

Hierarchy: Bold weights (600+) for headings, 400-500 for body text

Grid & Spacing:

8px base unit system (multiples of 8)

Follows mathematical precision principle of brand

React Dashboard Implementation Best Practices
Based on Polish banking sector trends (Credit Agricole CA24, mServices insights):

1. Component Architecture

Use modular, reusable components

Implement design tokens system

Support responsive layouts (mobile-first)

2. Key Dashboard Patterns (from Polish banking UI research):

Smart Accordion: Important products with progressive disclosure

Dashboard Widgets: KPI metrics, financial summaries

Smart Tables: Data visualization with responsiveness

Navigation: Clean vertical or horizontal navigation

Cards System: Data containers with hierarchy

3. Styling Approach for React

jsx
// Option A: CSS-in-JS with styled-components (Polish dev practice)
import styled from 'styled-components';

const PkoButton = styled.button`
  background-color: ${props => props.primary ? '#d93026' : '#1a2f3a'};
  color: white;
  padding: 8px 16px;
  border-radius: 4px;
  font-weight: 600;
  transition: all 0.3s ease;
  
  &:hover {
    opacity: 0.9;
    transform: translateY(-2px);
  }
`;

// Option B: Tailwind CSS (increasingly popular in Poland)
// With custom PKO theme extension

// Option C: CSS Modules with design tokens
import styles from './Dashboard.module.css';
4. Design Tokens Setup

javascript
const pkoTokens = {
  colors: {
    primary: '#1a2f3a',
    accent: '#d93026',
    gold: '#c9a961',
  },
  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '24px',
  },
  typography: {
    fontSize: {
      body: '14px',
      heading1: '28px',
      heading2: '24px',
    },
    fontWeight: {
      regular: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    }
  }
};
Accessibility & Trust Elements
Polish banking sector emphasizes:

High contrast: Navy + white/light backgrounds

Clear hierarchy: Information prioritization

Trust indicators: Professional appearance, secure interactions

Responsive design: Mobile-first approach

Loading states: Clear feedback during transactions

React Library Recommendations
Based on Polish developer practices:

Component Libraries:

CoreUI React (professional dashboards)

Shadcn/ui + Radix UI (accessible primitives)

TailGrids (Tailwind-based dashboard components)

Styling:

Tailwind CSS (growing adoption in Poland)

styled-components (established Polish practice)

CSS Modules (traditional approach)

Data Visualization:

Recharts (React-native charts)

Chart.js (with react wrapper)

D3.js (complex visualizations)

State Management:

React Query (data fetching)

Redux Toolkit (complex state)

Zustand (lightweight alternative)

Implementation Strategy for PKO-Themed Dashboard
Phase 1: Design System

Create design tokens file

Build component library matching PKO brand

Implement CSS variables for theming

Phase 2: Layout & Navigation

Responsive header with PKO logo

Sidebar navigation (vertical, clean)

Breadcrumb navigation

Phase 3: Dashboard Widgets

Account summary cards

Transaction history tables

Financial metrics (KPIs)

Charts for account analysis

Phase 4: Forms & Interactions

Form components with brand styling

Validation with clear messaging

Interactive elements with proper feedback

Current Polish Banking UX Trends (2024-2025)
API-first architecture (PKO BP strategy)

Cloud-native solutions (Google Cloud, cloud migration)

Agile development (time-to-market reduction)

Smart information disclosure (progressive UI)

Mobile-first responsive design

Strong security indicators (visual trust elements)

Would you like me to create a working React dashboard component that implements PKO BP branding guidelines? I can build:

A complete themed dashboard starter

Reusable component library matching PKO colors/typography

Specific dashboard widgets (cards, charts, tables)

Custom CSS or Tailwind theme configuration

Let me know which aspect you'd like to focus on for your Power Platform BI integration work!