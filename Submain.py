import pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import collections  # Для подсчёта дубликатов
import time  # Для логирования времени


# Функция для обработки дубликатов
def handle_duplicates(input_list, column_name, log_func=print):
    log_func(f"Проверка дубликатов в {column_name}...")
    counter = collections.Counter(input_list)
    new_list = []
    indices = {}
    for item in input_list:
        if counter[item] > 1:
            if item not in indices:
                indices[item] = 1
            new_item = f"{item}_{indices[item]}"
            indices[item] += 1
            new_list.append(new_item)
        else:
            new_list.append(item)
    log_func(f"Обработано дубликатов: {sum(1 for count in counter.values() if count > 1)}.")
    return new_list


# Оптимизированная фильтрация по Этап и Раздел
def filter_by_etap_razdel(sec_etap, sec_razdel, main_list_etap, main_list_razdel, fuzzy_threshold, embeddings_threshold,
                          emb_main_etap=None, emb_main_razdel=None, emb_sec_etap=None, emb_sec_razdel=None,
                          use_embeddings=False, use_fuzzy=True, used_main_indices=set()):
    filtered_indices = []

    # Fuzzy батч только если включен
    if use_fuzzy:
        etap_matches = process.extract(sec_etap, main_list_etap, scorer=fuzz.token_sort_ratio, limit=None)
        razdel_matches = process.extract(sec_razdel, main_list_razdel, scorer=fuzz.token_sort_ratio, limit=None)

        etap_dict = {idx: score for _, score, idx in etap_matches if score >= fuzzy_threshold}
        razdel_dict = {idx: score for _, score, idx in razdel_matches if score >= fuzzy_threshold}

        for idx in set(etap_dict.keys()) & set(razdel_dict.keys()):
            if idx not in used_main_indices:
                filtered_indices.append(idx)

    # Embeddings батч, если нужно (теперь работает независимо от fuzzy)
    if use_embeddings and emb_main_etap is not None and emb_main_razdel is not None and (
            not filtered_indices or not use_fuzzy):
        cos_etap = util.cos_sim(emb_sec_etap, emb_main_etap)[0]
        cos_razdel = util.cos_sim(emb_sec_razdel, emb_main_razdel)[0]
        for idx in range(len(main_list_etap)):
            if idx in used_main_indices:
                continue
            if cos_etap[idx] >= embeddings_threshold and cos_razdel[idx] >= embeddings_threshold:
                filtered_indices.append(idx)

    return filtered_indices


# Fuzzy matching с опцией отключения
def fuzz_match_with_order(main_df, secondary_df, fuzzy_threshold, fuzzy_etap_threshold, progress_bar, log_func, root,
                          model, use_embeddings, use_fuzzy, embeddings_etap_threshold, emb_main_etap, emb_main_razdel,
                          emb_main_names):
    main_list_names = handle_duplicates(main_df['Наименование работ'].astype(str).tolist(), "Наименование (главный)",
                                        log_func)
    main_list_etap = main_df['Этап'].astype(str).tolist()
    main_list_razdel = main_df['Раздел работ'].astype(str).tolist()
    main_list_n = main_df['N'].astype(str).tolist()

    sec_list_names = handle_duplicates(secondary_df['Наименование работ'].astype(str).tolist(),
                                       "Наименование (второстепенный)", log_func)
    sec_list_etap = secondary_df['Этап'].astype(str).tolist()
    sec_list_razdel = secondary_df['Раздел работ'].astype(str).tolist()

    total_rows = len(secondary_df)
    failed_indices = []
    used_main_indices = set()
    matched_n = [''] * total_rows
    match_info = [''] * total_rows

    progress_bar['maximum'] = total_rows

    # Если fuzzy выключен, все индексы переходят в failed для обработки embeddings
    if not use_fuzzy:
        failed_indices = list(range(total_rows))
        log_func("Fuzzy matching отключен, переход к embeddings...")
        return matched_n, failed_indices, used_main_indices, main_list_names, match_info

    log_func("Выполняется Fuzzy matching...")

    for sec_idx in range(total_rows):
        start_time = time.time()
        sec_etap = sec_list_etap[sec_idx].strip()
        sec_razdel = sec_list_razdel[sec_idx].strip()
        sec_name = sec_list_names[sec_idx].strip()

        if not sec_name or not sec_etap or not sec_razdel:
            failed_indices.append(sec_idx)
            continue

        emb_sec_etap = model.encode([sec_etap], convert_to_tensor=True)[0] if use_embeddings else None
        emb_sec_razdel = model.encode([sec_razdel], convert_to_tensor=True)[0] if use_embeddings else None

        filtered_indices = filter_by_etap_razdel(
            sec_etap, sec_razdel, main_list_etap, main_list_razdel, fuzzy_etap_threshold, embeddings_etap_threshold,
            emb_main_etap, emb_main_razdel, emb_sec_etap, emb_sec_razdel, use_embeddings, use_fuzzy, used_main_indices
        )

        if not filtered_indices:
            failed_indices.append(sec_idx)
            continue

        available_names = [main_list_names[i] for i in filtered_indices]

        match = process.extractOne(sec_name, available_names, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= fuzzy_threshold:
            matched_name = match[0]
            matched_idx = filtered_indices[available_names.index(matched_name)]
            matched_n[sec_idx] = main_list_n[matched_idx]
            match_info[sec_idx] = f"Fuzzy: {match[1]}%"
            used_main_indices.add(matched_idx)
        else:
            failed_indices.append(sec_idx)

        progress_bar['value'] = sec_idx + 1
        root.update_idletasks()
        log_func(f"Строка {sec_idx + 1}/{total_rows} обработана (время: {time.time() - start_time:.2f} сек).")

    return matched_n, failed_indices, used_main_indices, main_list_names, match_info


# Embeddings уточнение с поддержкой работы без fuzzy
def embeddings_refine_for_failed(failed_indices, main_df, secondary_df, main_list_names, model, embeddings_threshold,
                                 fuzzy_etap_threshold, embeddings_etap_threshold, used_main_indices, log_func,
                                 use_embeddings, use_fuzzy, emb_main_etap, emb_main_razdel, emb_main_names):
    matched_n = [''] * len(secondary_df)
    match_info = [''] * len(secondary_df)

    sec_list_names = secondary_df['Наименование работ'].astype(str).tolist()
    sec_list_etap = secondary_df['Этап'].astype(str).tolist()
    sec_list_razdel = secondary_df['Раздел работ'].astype(str).tolist()
    main_list_etap = main_df['Этап'].astype(str).tolist()
    main_list_razdel = main_df['Раздел работ'].astype(str).tolist()
    main_list_n = main_df['N'].astype(str).tolist()

    for sec_idx in failed_indices:
        start_time = time.time()
        sec_etap = sec_list_etap[sec_idx].strip()
        sec_razdel = sec_list_razdel[sec_idx].strip()
        sec_name = sec_list_names[sec_idx].strip()

        if not sec_name:
            continue

        emb_sec_etap = model.encode([sec_etap], convert_to_tensor=True)[0] if use_embeddings else None
        emb_sec_razdel = model.encode([sec_razdel], convert_to_tensor=True)[0] if use_embeddings else None

        filtered_indices = filter_by_etap_razdel(
            sec_etap, sec_razdel, main_list_etap, main_list_razdel, fuzzy_etap_threshold, embeddings_etap_threshold,
            emb_main_etap, emb_main_razdel, emb_sec_etap, emb_sec_razdel, use_embeddings, use_fuzzy, used_main_indices
        )

        if not filtered_indices:
            continue

        emb_b = model.encode([sec_name], convert_to_tensor=True)[0]
        filtered_emb = emb_main_names[torch.tensor(filtered_indices)]
        cos_scores = util.cos_sim(emb_b, filtered_emb)[0]

        if cos_scores.max() < embeddings_threshold:
            continue

        max_idx = cos_scores.argmax().item()
        matched_main_idx = filtered_indices[max_idx]
        max_score = cos_scores.max().item()

        matched_n[sec_idx] = main_list_n[matched_main_idx]
        match_info[sec_idx] = f"Embeddings: {max_score * 100:.1f}%"
        used_main_indices.add(matched_main_idx)

        log_func(
            f"Embeddings: Строка {sec_idx + 1} обработана (сходство: {max_score:.2f}, время: {time.time() - start_time:.2f} сек).")

    return matched_n, match_info


# Основная функция с поддержкой отключения fuzzy и новой опции для дубликатов
def process_data(input_file, fuzzy_threshold, embeddings_threshold, fuzzy_etap_threshold, embeddings_etap_threshold,
                 use_embeddings, use_fuzzy, allow_duplicate_n, progress_bar, log_text, root):
    try:
        def log_func(msg):
            log_text.insert(tk.END, msg + '\n')
            log_text.see(tk.END)
            root.update_idletasks()

        # Проверка что хотя бы один метод включен
        if not use_fuzzy and not use_embeddings:
            messagebox.showerror("Ошибка",
                                 "Должен быть включен хотя бы один метод сопоставления (Fuzzy или Embeddings)!")
            return

        log_func(
            f"Используемые методы: {'Fuzzy' if use_fuzzy else ''}{'+ ' if use_fuzzy and use_embeddings else ''}{'Embeddings' if use_embeddings else ''}")
        log_func(f"Одинаковый N для дубликатов: {'ВКЛЮЧЕНО' if allow_duplicate_n else 'ВЫКЛЮЧЕНО'}")

        # Используем контекстный менеджер для правильного закрытия файла
        with pd.ExcelFile(input_file) as excel_file:
            sheet_names = excel_file.sheet_names

            if "0" not in sheet_names:
                messagebox.showerror("Ошибка", "Лист '0' не найден!")
                return

            # Читаем все необходимые данные сразу и сохраняем в словарь
            all_sheets_data = {}

            # Читаем главный лист
            main_df = pd.read_excel(excel_file, sheet_name="0")
            required_cols = ["N", "Наименование работ", "Этап", "Раздел работ"]
            if not all(col in main_df.columns for col in required_cols):
                messagebox.showerror("Ошибка", f"Отсутствуют столбцы: {required_cols}")
                return

            main_df['N'] = main_df['N'].astype(str)

            # Читаем все второстепенные листы
            for sheet in sheet_names:
                if sheet != "0":
                    sheet_df = pd.read_excel(excel_file, sheet_name=sheet)
                    sheet_df['N'] = sheet_df['N'].astype(str)
                    if 'Тип сопоставления' not in sheet_df.columns:
                        sheet_df['Тип сопоставления'] = ''
                    all_sheets_data[sheet] = sheet_df

            log_func("Все листы загружены в память. Файл Excel закрыт.")

        # Теперь файл закрыт, и мы можем работать с данными в памяти
        writer = pd.ExcelWriter('output_updated.xlsx', engine='openpyxl')
        main_df.to_excel(writer, sheet_name="0", index=False)

        total_rows_all = sum(len(df) for df in all_sheets_data.values())
        progress_bar['maximum'] = total_rows_all
        total_progress = 0

        model = SentenceTransformer('distiluse-base-multilingual-cased-v1') if use_embeddings else None

        # Предвычисление эмбеддингов для главного листа
        emb_main_etap = model.encode(main_df['Этап'].astype(str).tolist(),
                                     convert_to_tensor=True) if use_embeddings else None
        emb_main_razdel = model.encode(main_df['Раздел работ'].astype(str).tolist(),
                                       convert_to_tensor=True) if use_embeddings else None
        emb_main_names = model.encode(main_df['Наименование работ'].astype(str).tolist(),
                                      convert_to_tensor=True) if use_embeddings else None
        log_func("Эмбеддинги главного листа предвычислены (если включены).")

        global_used_count = collections.Counter()

        for sheet, secondary_df in all_sheets_data.items():
            log_func(f"Обработка листа: {sheet}")

            if not all(col in secondary_df.columns for col in required_cols):
                log_func(f"Предупреждение: Лист '{sheet}' пропущен.")
                secondary_df.to_excel(writer, sheet_name=sheet, index=False)
                continue

            progress_bar['maximum'] = len(secondary_df)
            progress_bar['value'] = 0

            # ДОБАВЛЕНО: Обработка дубликатов по опции
            if allow_duplicate_n:
                # Создаем уникальный датафрейм по ключу
                key_cols = ['Наименование работ', 'Этап', 'Раздел работ']
                df_unique = secondary_df.drop_duplicates(subset=key_cols).copy()
                df_unique['original_indices'] = df_unique.index  # Сохраняем оригинальные индексы для маппинга
                total_unique = len(df_unique)
                log_func(
                    f"Обнаружено {len(secondary_df) - total_unique} дубликатов. Matching на {total_unique} уникальных комбинациях.")

                # Matching на уникальном df
                fuzzy_matched_n, failed_indices, used_main_indices, main_list_names, fuzzy_match_info = fuzz_match_with_order(
                    main_df, df_unique, fuzzy_threshold, fuzzy_etap_threshold, progress_bar, log_func, root, model,
                    use_embeddings, use_fuzzy, embeddings_etap_threshold, emb_main_etap, emb_main_razdel, emb_main_names
                )

                embeddings_matched_n = [''] * len(df_unique)
                embeddings_match_info = [''] * len(df_unique)

                if use_embeddings and failed_indices:
                    embeddings_matched_n, embeddings_match_info = embeddings_refine_for_failed(
                        failed_indices, main_df, df_unique, main_list_names, model, embeddings_threshold,
                        fuzzy_etap_threshold, embeddings_etap_threshold, used_main_indices, log_func, use_embeddings,
                        use_fuzzy, emb_main_etap, emb_main_razdel, emb_main_names
                    )

                # Создаем словарь маппинга: ключ -> (N, info)
                match_map = {}
                for i in range(len(df_unique)):
                    key = tuple(df_unique.iloc[i][key_cols])
                    n_val = fuzzy_matched_n[i] if fuzzy_matched_n[i] else embeddings_matched_n[i]
                    info_val = fuzzy_match_info[i] if fuzzy_match_info[i] else embeddings_match_info[i]
                    match_map[key] = (str(n_val) if n_val else '', info_val if info_val else 'Не найдено')

                # Присваиваем N и info всем строкам в полном df (включая дубликаты)
                for i in range(len(secondary_df)):
                    key = tuple(secondary_df.iloc[i][key_cols])
                    if key in match_map:
                        n_val, info_val = match_map[key]
                        secondary_df.at[i, 'N'] = n_val
                        secondary_df.at[i, 'Тип сопоставления'] = info_val
                    else:
                        secondary_df.at[i, 'Тип сопоставления'] = 'Не найдено'
            else:
                # Стандартный режим без обработки дубликатов
                fuzzy_matched_n, failed_indices, used_main_indices, main_list_names, fuzzy_match_info = fuzz_match_with_order(
                    main_df, secondary_df, fuzzy_threshold, fuzzy_etap_threshold, progress_bar, log_func, root, model,
                    use_embeddings, use_fuzzy, embeddings_etap_threshold, emb_main_etap, emb_main_razdel, emb_main_names
                )

                embeddings_matched_n = [''] * len(secondary_df)
                embeddings_match_info = [''] * len(secondary_df)

                if use_embeddings and failed_indices:
                    embeddings_matched_n, embeddings_match_info = embeddings_refine_for_failed(
                        failed_indices, main_df, secondary_df, main_list_names, model, embeddings_threshold,
                        fuzzy_etap_threshold, embeddings_etap_threshold, used_main_indices, log_func, use_embeddings,
                        use_fuzzy, emb_main_etap, emb_main_razdel, emb_main_names
                    )

                for i in range(len(secondary_df)):
                    if fuzzy_matched_n[i]:
                        secondary_df.at[i, 'N'] = str(fuzzy_matched_n[i])
                        secondary_df.at[i, 'Тип сопоставления'] = fuzzy_match_info[i]
                    elif embeddings_matched_n[i]:
                        secondary_df.at[i, 'N'] = str(embeddings_matched_n[i])
                        secondary_df.at[i, 'Тип сопоставления'] = embeddings_match_info[i]
                    else:
                        secondary_df.at[i, 'Тип сопоставления'] = 'Не найдено'

            for idx in used_main_indices:
                global_used_count[idx] += 1

            secondary_df.to_excel(writer, sheet_name=sheet, index=False)

            total_progress += len(secondary_df)
            progress_bar['maximum'] = total_rows_all
            progress_bar['value'] = total_progress
            root.update_idletasks()

            matched_count = len(secondary_df[secondary_df['Тип сопоставления'] != 'Не найдено'])
            unsop = len(secondary_df) - matched_count
            log_func(
                f"Лист '{sheet}' обработан. Сопоставлено: {matched_count}, Несопоставлено: {unsop}. Уникальных из главного: {len(used_main_indices)}.")

        writer.close()

        log_func("Глобальное использование строк главного:")
        for idx, count in global_used_count.items():
            if count > 1:
                log_func(f"Строка {idx + 1} использована {count} раз (в разных листах).")

        messagebox.showinfo("Успех",
                            "Готово! Файл: output_updated.xlsx\n\nДобавлен столбец 'Тип сопоставления' с информацией о методе и проценте совпадения.")

    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка: {e}")


# GUI с добавлением чекбокса для дубликатов
def create_gui():
    root = tk.Tk()
    root.title("Сопоставление работ с опцией для дубликатов")
    root.geometry("600x850")

    # ПРЕДУПРЕЖДЕНИЕ
    warning_frame = tk.Frame(root, bg='yellow')
    warning_frame.pack(fill='x', pady=5)
    tk.Label(warning_frame, text="⚠️ ВАЖНО: Закройте Excel-файл перед запуском скрипта!",
             bg='yellow', fg='red', font=('Arial', 10, 'bold')).pack(pady=5)


    tk.Label(root, text="Входной файл Excel:").pack(pady=5)
    file_entry = tk.Entry(root, width=50)
    file_entry.pack()
    file_entry.insert(0, "input.xlsx")

    def browse_file():
        file = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file:
            file_entry.delete(0, tk.END)
            file_entry.insert(0, file)

    tk.Button(root, text="Выбрать файл", command=browse_file).pack(pady=5)

    tk.Label(root, text="Fuzzy threshold для Наименования (65-90):").pack(pady=5)
    fuzzy_entry = tk.Entry(root)
    fuzzy_entry.pack()
    fuzzy_entry.insert(0, "65")

    tk.Label(root, text="Fuzzy threshold для Этап/Раздел (80-95):").pack(pady=5)
    fuzzy_etap_entry = tk.Entry(root)
    fuzzy_etap_entry.pack()
    fuzzy_etap_entry.insert(0, "80")

    tk.Label(root, text="Embeddings threshold для Наименования (0.5-0.9):").pack(pady=5)
    embeddings_entry = tk.Entry(root)
    embeddings_entry.pack()
    embeddings_entry.insert(0, "0.7")

    tk.Label(root, text="Embeddings threshold для Этап/Раздел (0.8-0.95):").pack(pady=5)
    embeddings_etap_entry = tk.Entry(root)
    embeddings_etap_entry.pack()
    embeddings_etap_entry.insert(0, "0.8")

    # Чекбоксы для методов
    methods_frame = tk.Frame(root)
    methods_frame.pack(pady=10)

    use_fuzzy_var = tk.BooleanVar(value=True)
    tk.Checkbutton(methods_frame, text="Использовать Fuzzy matching", variable=use_fuzzy_var,
                   font=('Arial', 10, 'bold'), fg='green').pack(anchor='w')

    use_embeddings_var = tk.BooleanVar(value=True)
    tk.Checkbutton(methods_frame, text="Использовать эмбеддинги", variable=use_embeddings_var,
                   font=('Arial', 10, 'bold'), fg='blue').pack(anchor='w')

    # ДОБАВЛЕНО: Чекбокс для дубликатов
    allow_duplicate_n_var = tk.BooleanVar(value=False)
    tk.Checkbutton(methods_frame, text="Разрешить одинаковый N для дубликатов", variable=allow_duplicate_n_var,
                   font=('Arial', 10, 'bold'), fg='purple').pack(anchor='w')

    tk.Label(root, text="Прогресс:").pack(pady=5)
    progress = Progressbar(root, maximum=100, length=300)
    progress.pack(pady=5)

    tk.Label(root, text="Логи:").pack(pady=5)
    log_text = tk.Text(root, height=8, width=60)
    log_text.pack(pady=5)

    def run_process():
        input_file = file_entry.get()
        try:
            fuzzy_threshold = int(fuzzy_entry.get())
            fuzzy_etap_threshold = int(fuzzy_etap_entry.get())
            embeddings_threshold = float(embeddings_entry.get())
            embeddings_etap_threshold = float(embeddings_etap_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Пороги должны быть числами!")
            return

        # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА
        try:
            test_file = pd.ExcelFile(input_file)
            test_file.close()
        except Exception as e:
            messagebox.showerror("Ошибка доступа к файлу",
                                 f"Не удается открыть файл. Убедитесь, что:\n"
                                 f"1. Файл не открыт в Excel\n"
                                 f"2. Путь к файлу корректен\n"
                                 f"3. У вас есть права на чтение файла\n\n"
                                 f"Ошибка: {e}")
            return

        log_text.delete(1.0, tk.END)
        progress['value'] = 0
        process_data(input_file, fuzzy_threshold, embeddings_threshold, fuzzy_etap_threshold, embeddings_etap_threshold,
                     use_embeddings_var.get(), use_fuzzy_var.get(), allow_duplicate_n_var.get(), progress, log_text,
                     root)

    tk.Button(root, text="Запустить", command=run_process).pack(pady=10)
    tk.Button(root, text="Выход", command=root.quit).pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    create_gui()
