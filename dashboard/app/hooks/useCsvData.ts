import { useState, useEffect } from "react";
import Papa from "papaparse";

export function useCsvData<T>(url: string, options: Papa.ParseConfig = {}) {
  const [data, setData] = useState<T[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    Papa.parse(url, {
      download: true,
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      ...options,
      complete: (results) => {
        setData(results.data as T[]);
        setLoading(false);
        if (options.complete) {
            options.complete(results, undefined as any);
        }
      },
      error: (err, file) => {
        setError(err.message);
        setLoading(false);
        if (options.error) {
            options.error(err, file);
        }
      },
    });
  }, [url]); // We can't easily add options to dependency array if it's an object literal created on every render

  return { data, loading, error };
}
