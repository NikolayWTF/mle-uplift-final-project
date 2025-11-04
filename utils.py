import matplotlib.pyplot as plt
import numpy as np
from sklift.metrics import uplift_by_percentile
from scipy import stats
import pandas as pd
from sklift.viz import plot_qini_curve, plot_uplift_curve
from sklift.metrics import uplift_auc_score, qini_auc_score, uplift_at_k

def custom_uplift_by_percentile(y_true, uplift, treatment, 
                               kind='line', bins=10, string_percentiles=True, 
                               figsize=(10, 6), title=None):
    """
    Построение графика uplift по перцентилям.
    
    Аргументы:
        y_true: Бинарные целевые значения
        uplift: Прогнозируемые значения uplift
        treatment: Бинарные индикаторы воздействия
        kind: 'line' или 'bar'
        bins: Количество перцентильных корзин
        string_percentiles: Отображать ли перцентили в виде строк
        figsize: Размер рисунка (кортеж)
        title: Пользовательский заголовок для графика
    
    Возвращает:
        Рисунок matplotlib
    """
    
    # получаем данные по перцентилям, используя функцию из sklift
    df = uplift_by_percentile(
        y_true, uplift, treatment, strategy='overall',
        std=True, total=False, bins=bins, string_percentiles=False
    )
    
    # извлекаем перцентили из индекса DataFrame
    percentiles = df.index[:bins].values.astype(float)
    
    # извлекаем значения отклика для тестовой группы и их стандартные отклонения
    response_rate_trmnt = df.loc[percentiles, 'response_rate_treatment'].values
    std_trmnt = df.loc[percentiles, 'std_treatment'].values
    
    # извлекаем значения отклика для контрольной группы и их стандартные отклонения
    response_rate_ctrl = df.loc[percentiles, 'response_rate_control'].values
    std_ctrl = df.loc[percentiles, 'std_control'].values
    
    # извлекаем значения uplift и их стандартные отклонения
    uplift_score = df.loc[percentiles, 'uplift'].values
    std_uplift = df.loc[percentiles, 'std_uplift'].values
    
    # создаём график
    fig, ax = plt.subplots(figsize=figsize)
    
    if kind == 'line':
        # строим линейный график для тестовой группы с погрешностями
        ax.errorbar(
            percentiles, response_rate_trmnt, yerr=std_trmnt,
            linewidth=2, color='forestgreen', label='Отклик тестовой группы'
        )
        # строим линейный график для контрольной группы с погрешностями
        ax.errorbar(
            percentiles, response_rate_ctrl, yerr=std_ctrl,
            linewidth=2, color='orange', label='Отклик контрольной группы'
        )
        # строим линейный график для uplift с погрешностями
        ax.errorbar(
            percentiles, uplift_score, yerr=std_uplift,
            linewidth=2, color='red', label='Uplift'
        )
        # заполняем область между линиями тестовой и контрольной групп
        ax.fill_between(percentiles, response_rate_trmnt,
                        response_rate_ctrl, alpha=0.1, color='red')
        
        # добавляем горизонтальную линию на уровне 0, если есть отрицательные значения uplift
        if np.amin(uplift_score) < 0:
            ax.axhline(y=0, color='black', linewidth=1)
            
    elif kind == 'bar':
        # вычисляем ширину столбцов для столбчатой диаграммы
        width = percentiles[1] - percentiles[0] if len(percentiles) > 1 else 5
        bar_width = width * 0.35
        
        # строим столбцы для тестовой, контрольной групп и для uplift
        ax.bar(percentiles - bar_width, response_rate_trmnt, bar_width, 
               color='forestgreen', label='Отклик тестовой группы')
        ax.bar(percentiles, response_rate_ctrl, bar_width, 
               color='orange', label='Отклик контрольной группы')
        ax.bar(percentiles + bar_width, uplift_score, bar_width, 
               color='red', label='Uplift')
    
    # устанавливаем метки по оси X
    if string_percentiles:
        # создаём строковые метки для перцентилей (диапазоны)
        percentiles_str = [f"0-{percentiles[0]:.0f}"] + \
                          [f"{percentiles[i]:.0f}-{percentiles[i + 1]:.0f}" 
                           for i in range(len(percentiles) - 1)]
        ax.set_xticks(percentiles)
        ax.set_xticklabels(percentiles_str, rotation=45)
    else:
        # используем числовые значения перцентилей
        ax.set_xticks(percentiles)
    
    # устанавливаем подписи осей и заголовок
    ax.set_xlabel('Перцентиль')
    ax.set_ylabel('Уровень отклика / Uplift')
    
    # устанавливаем заголовок, если он предоставлен
    if title:
        ax.set_title(title)
  
    # добавляем легенду и сетку для улучшения читаемости
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # оптимизируем расположение элементов на графике
    plt.tight_layout()
    return fig

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - (k - 1)*(r - 1)/(n - 1))
    rcorr = r - (r - 1)**2/(n - 1)
    kcorr = k - (k - 1)**2/(n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

def eta_squared(y, x):
    try:
        groups = [y[x == cat] for cat in np.unique(x)]
        f_val, p_val = stats.f_oneway(*groups)
        ss_between = sum(len(g) * (g.mean() - y.mean())**2 for g in groups)
        ss_total = sum((y - y.mean())**2)
        return ss_between / ss_total if ss_total != 0 else 0
    except Exception:
        return np.nan

def plot_uplift_results(y_true, uplift_pred, treatment, k=0.3):
    """
    Визуализирует Qini и Uplift кривые и выводит ключевые метрики uplift-модели.
    
    Параметры
    ----------
    y_true : array-like
        Фактические значения целевой переменной.
    uplift_pred : array-like
        Предсказанные значения uplift (модельный uplift).
    treatment : array-like
        Бинарный индикатор treatment-группы (1 — treatment, 0 — control).
    k : float, optional (default=0.3)
        Доля топ-N% клиентов для расчёта метрики uplift@k.
    """
    
    # --- Графики ---
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Qini Curve
    plot_qini_curve(
        y_true,
        uplift_pred,
        treatment,
        perfect=True,
        ax=axs[0],
        name='Qini Curve'
    )
    axs[0].set_title("Qini Curve")

    # Uplift Curve
    plot_uplift_curve(
        y_true,
        uplift_pred,
        treatment,
        perfect=True,
        ax=axs[1],
        name='Uplift Curve'
    )
    axs[1].set_title("Uplift Curve")

    plt.tight_layout()
    plt.show()

    # --- Метрики ---
    qini = qini_auc_score(y_true=y_true, uplift=uplift_pred, treatment=treatment)
    uplift_auc = uplift_auc_score(y_true=y_true, uplift=uplift_pred, treatment=treatment)
    uplift_topk = uplift_at_k(y_true=y_true, uplift=uplift_pred, treatment=treatment, strategy='by_group', k=k)

    print("📊 Метрики модели:")
    print(f"Qini AUC:    {qini:.4f}")
    print(f"Uplift AUC:  {uplift_auc:.4f}")
    print(f"Uplift@{int(k*100)}%:  {uplift_topk:.4f}")

    return {
        "qini_auc": qini,
        "uplift_auc": uplift_auc,
        f"uplift@{int(k*100)}%": uplift_topk
    }