import pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import collections  # Для подсчёта дубликатов


# Функция для обработки дубликатов в списке (добавляем суффиксы _1, _2 и т.д. для повторяющихся)
def handle_duplicates(input_list, column_name, log_func=print):
    log_func(f"Проверка и обработка дубликатов в столбце {column_name}...")
    counter = collections.Counter(input_list)  # Подсчёт частоты каждого значения
    new_list = []
    indices = {}  # Словарь для отслеживания индексов дубликатов (независимый для каждого столбца)

    for item in input_list:
        if counter[item] > 1:  # Если дубликат
            if item not in indices:
                indices[item] = 1
            new_item = f"{item}_{indices[item]}"
            indices[item] += 1
            new_list.append(new_item)
        else:
            new_list.append(item)

    log_func(
        f"Обработано дубликатов в {column_name}: {sum(1 for count in counter.values() if count > 1)} значений получили индексы.")
    return new_list


# Функция для fuzzy matching с one-to-one
def fuzz_match_with_order(df, col1, col2, threshold=65, progress_bar=None, log_func=print, root=None):
    list_a_original = df[col1].astype(str).tolist()
    list_b_original = df[col2].astype(str).tolist()

    # Обработка дубликатов независимо для A и B перед сопоставлением
    list_a = handle_duplicates(list_a_original, "A", log_func)
    list_b = handle_duplicates(list_b_original, "B", log_func)

    total_rows = len(df)
    failed_indices = []
    used_a = set()  # Для one-to-one (только для A)
    results = []

    if progress_bar:
        progress_bar['maximum'] = total_rows
    log_func(f"Начинаем fuzzy matching для {total_rows} строк...")

    for index in range(total_rows):
        item_b = list_b[index] if list_b[index].strip() else ""
        candidate = None
        sim = 0
        if item_b:
            available_a = [a for a in list_a if a not in used_a]
            match = process.extractOne(item_b, available_a, scorer=fuzz.token_sort_ratio)
            if match and match[1] >= threshold:
                candidate = match[0]
                sim = match[1]
                used_a.add(candidate)  # Помечаем как использованное
            else:
                failed_indices.append(index)

        results.append({
            'Исходный индекс': index + 1,
            'Первый столбец (A)': list_a[index] if index < len(list_a) else "",  # Обновлённое A с индексами
            'Второй столбец (B)': list_b[index] if index < len(list_b) else "",  # Обновлённое B с индексами
            'Кандидат из fuzzy': candidate,
            'Fuzzy сходство': sim,
            'Кандидат из векторов': None,
            'Сходство векторов (%)': 0
        })

        if progress_bar and root:
            progress_bar['value'] = index + 1
            root.update_idletasks()
        log_func(f"Fuzzy: Обработана строка {index + 1}/{total_rows}")

    log_func("Fuzzy matching завершён.")
    return results, failed_indices, list_a, list_b, used_a


# Функция для embeddings только для не прошедших fuzzy
def embeddings_refine_for_failed(failed_indices, list_a, list_b, model, threshold=0.7, used_a=set(), log_func=print):
    embeddings_a = model.encode(list_a, convert_to_tensor=True)
    results = [None] * len(list_b)
    sim_scores = [0] * len(list_b)

    log_func(f"Вычисление эмбеддингов для {len(failed_indices)} не прошедших fuzzy строк...")

    for idx in failed_indices:
        item_b = list_b[idx]
        emb_b = model.encode([item_b], convert_to_tensor=True)[0]
        cos_scores = util.cos_sim(emb_b, embeddings_a)[0]
        # Маскируем использованные
        for j, item in enumerate(list_a):
            if item in used_a:
                cos_scores[j] = -1
        if cos_scores.max() < 0:
            continue
        max_sim_index = cos_scores.argmax().item()
        max_sim_score = cos_scores[max_sim_index].item()

        if max_sim_score >= threshold:
            candidate = list_a[max_sim_index]
            results[idx] = candidate
            sim_scores[idx] = max_sim_score * 100
            used_a.add(candidate)  # Помечаем как использованное
        log_func(f"Embeddings: Обработана строка {idx + 1} (сходство: {max_sim_score:.2f})")

    log_func("Эмбеддинги завершены.")
    return results, sim_scores


# Основная функция обработки
def process_data(input_file, fuzzy_threshold, embeddings_threshold, use_embeddings, progress_bar, log_text, root):
    try:
        df = pd.read_excel(input_file)
        if df.empty:
            messagebox.showerror("Ошибка", "Файл пустой!")
            return

        col1, col2 = df.columns[:2]

        def log_func(msg):
            log_text.insert(tk.END, msg + '\n')
            log_text.see(tk.END)
            root.update_idletasks()

        # Шаг 1: Fuzzy
        progress_bar['value'] = 0
        fuzzy_results, failed_indices, list_a, list_b, used_a = fuzz_match_with_order(df, col1, col2, fuzzy_threshold,
                                                                                      progress_bar, log_func, root)

        # Шаг 2: Embeddings, если включены и есть failed
        if use_embeddings and failed_indices:
            model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
            progress_bar['value'] = 0
            progress_bar['maximum'] = len(failed_indices)
            embeddings_candidates, embeddings_sims = embeddings_refine_for_failed(failed_indices, list_a, list_b, model,
                                                                                  embeddings_threshold, used_a,
                                                                                  log_func)

            # Обновляем результаты
            for i, res in enumerate(fuzzy_results):
                if embeddings_candidates[i]:
                    res['Кандидат из векторов'] = embeddings_candidates[i]
                    res['Сходство векторов (%)'] = embeddings_sims[i]

        # Финальные результаты
        final_results = []
        unmatched_b_count = 0
        for pair in fuzzy_results:
            if pair['Кандидат из fuzzy'] or pair['Кандидат из векторов']:
                final_results.append({**pair, 'Вердикт': "Сопоставлено"})
            else:
                final_results.append({**pair, 'Вердикт': "Несопоставлено"})
                unmatched_b_count += 1

        # Добавляем несопоставленные из A (с учётом индексов)
        all_a_unique = set(list_a)
        unmatched_a = list(all_a_unique - used_a)
        log_func(f"Добавляем {len(unmatched_a)} несопоставленных значений из A в конец.")
        for ua in unmatched_a:
            final_results.append({
                'Исходный индекс': None,
                'Первый столбец (A)': ua,
                'Второй столбец (B)': None,
                'Кандидат из fuzzy': None,
                'Fuzzy сходство': 0,
                'Кандидат из векторов': None,
                'Сходство векторов (%)': 0,
                'Вердикт': "Несопоставлено (висящее из A)"
            })

        log_func(f"Несопоставленных из B: {unmatched_b_count}")

        # Сохраняем
        result_df = pd.DataFrame(final_results)
        result_df.to_excel('output_matches.xlsx', index=False)
        messagebox.showinfo("Успех", "Готово! Результаты в output_matches.xlsx")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")


# GUI
def create_gui():
    root = tk.Tk()
    root.title("Сопоставление данных (Fuzzy + Embeddings)")
    root.geometry("500x700")

    tk.Label(root, text="Входной файл:").pack(pady=5)
    file_entry = tk.Entry(root, width=50)
    file_entry.pack()
    file_entry.insert(0, "input.xlsx")

    def browse_file():
        file = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file:
            file_entry.delete(0, tk.END)
            file_entry.insert(0, file)

    tk.Button(root, text="Выбрать файл", command=browse_file).pack(pady=5)

    tk.Label(root, text="Fuzzy threshold (65-90):").pack(pady=5)
    fuzzy_entry = tk.Entry(root)
    fuzzy_entry.pack()
    fuzzy_entry.insert(0, "71")

    tk.Label(root, text="Embeddings threshold (0.5-0.9):").pack(pady=5)
    embeddings_entry = tk.Entry(root)
    embeddings_entry.pack()
    embeddings_entry.insert(0, "0.7")

    use_embeddings_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="Использовать эмбеддинги (вектора)", variable=use_embeddings_var).pack(pady=5)

    tk.Label(root, text="Прогресс:").pack(pady=5)
    progress = Progressbar(root, maximum=100, length=300)
    progress.pack(pady=5)

    tk.Label(root, text="Логи:").pack(pady=5)
    log_text = tk.Text(root, height=10, width=60)
    log_text.pack(pady=5)

    def run_process():
        input_file = file_entry.get()
        try:
            fuzzy_threshold = int(fuzzy_entry.get())
            embeddings_threshold = float(embeddings_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Пороги должны быть числами!")
            return
        log_text.delete(1.0, tk.END)
        progress['value'] = 0
        process_data(input_file, fuzzy_threshold, embeddings_threshold, use_embeddings_var.get(), progress, log_text,
                     root)

    tk.Button(root, text="Запустить", command=run_process).pack(pady=10)
    tk.Button(root, text="Выход", command=root.quit).pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    create_gui()
