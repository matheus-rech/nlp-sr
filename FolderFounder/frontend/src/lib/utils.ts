import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function getConfidenceColor(confidence: number): string {
  if (confidence >= 80) return "text-green-600 dark:text-green-400"
  if (confidence >= 60) return "text-yellow-600 dark:text-yellow-400"
  return "text-red-600 dark:text-red-400"
}

export function formatDate(date: string | Date): string {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  }).format(new Date(date))
}