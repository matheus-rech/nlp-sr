const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api/v1';

// --- TypeScript interfaces for type safety ---
export interface Project {
  id: string;
  name: string;
  // Add other project fields as needed
}

export interface Citation {
  id: string;
  title: string;
  // Add other citation fields as needed
}

export interface ApiError {
  message: string;
  status?: number;
}

// --- Reusable fetch helper ---
async function fetchWithHandling<T>(
  input: RequestInfo,
  init?: RequestInit,
  isBlob?: boolean
): Promise<T> {
  try {
    const response = await fetch(input, init);
    if (!response.ok) {
      let errorMsg = `Error: ${response.status}`;
      try {
        const data = await response.json();
        errorMsg = data.message || errorMsg;
      } catch {}
      throw { message: errorMsg, status: response.status } as ApiError;
    }
    if (isBlob) {
      // @ts-ignore
      return response.blob();
    }
    return response.json();
  } catch (error: any) {
    if (error.message) throw error;
    throw { message: 'Network error', status: undefined } as ApiError;
  }
}

export const api = {
  // Projects
  createProject: async (data: Partial<Project>): Promise<Project> => {
    return fetchWithHandling<Project>(`${API_BASE_URL}/projects/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
  },

  getProjects: async (): Promise<Project[]> => {
    return fetchWithHandling<Project[]>(`${API_BASE_URL}/projects/`);
  },

  // Citations
  uploadCitations: async (projectId: string, file: File): Promise<Citation[]> => {
    const formData = new FormData();
    formData.append('file', file);
    return fetchWithHandling<Citation[]>(`${API_BASE_URL}/projects/${projectId}/citations/upload`, {
      method: 'POST',
      body: formData,
    });
  },

  getCitations: async (projectId: string): Promise<Citation[]> => {
    return fetchWithHandling<Citation[]>(`${API_BASE_URL}/projects/${projectId}/citations`);
  },

  // Screening
  startScreening: async (projectId: string): Promise<any> => {
    return fetchWithHandling<any>(`${API_BASE_URL}/projects/${projectId}/screen`, {
      method: 'POST',
    });
  },

  // Export
  exportResults: async (projectId: string, format: string = 'json'): Promise<Blob> => {
    return fetchWithHandling<Blob>(`${API_BASE_URL}/projects/${projectId}/export?format=${format}`, undefined, true);
  },
};
