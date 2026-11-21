# Improve Stage 3 target geometry with robust multi-ring projective fitting

Draft PR в `main` из `feature/robust-target-geometry`.

Повреждённые кольца и перспектива смещали центр, рассчитанный по эллипсу. Новый estimator совместно подгоняет гомографию по измеренным дугам нескольких известных радиусов, восстанавливает физический центр через обратную гомографию и отклоняет неоднозначные/неустойчивые решения. Legacy detector сохранён. Добавлены явный выбор группы/ROI, диагностика, внешняя ориентация, synthetic generator и CLI сравнения.

На одинаковых 58 фото медиана/P95 отклонения от пользовательских отметок для 171 парных центров снизилась 9.48/43.58→2.68/18.77px; один из 172 центров не восстановлен. READY снизилось 25→20, девять регрессий принятия подробно описаны в отчёте. Эти отметки — development feedback, независимый реальный holdout **NOT_VALIDATED**. На 27 независимых синтетических сценах READY 18→20, P95 центра READY .012962→.000573; 0/3 negative scenes приняты. Малый набор не доказывает production false-accept limit.

Проверки: 127 unit tests PASS без skips; Ruff format/check PASS; diff check PASS. Desktop 3000×4000 median/P95 legacy 721/946ms, robust 831/1030ms; peak RSS 596/696MiB. Android performance и CI не проверялись. Production validation policy не изменена.

Подробные результаты, ограничения, воспроизведение и происхождение данных: `docs/robust-target-geometry.md`, агрегаты `docs/reports/robust-geometry-comparison.json`. Личные фото и большие диагностические артефакты остаются локально. Требуется независимый holdout перед подтверждением production-quality.

По ревью исправлены false-accept denominator, missing previews и обязательная RGB-привязка обработанных входов; добавлены интеграционные регрессионные тесты. Повторный парный benchmark27+58 на78b93bc подтвердил прежние geometry-метрики. Исправленный агрегат: `docs/reports/robust-geometry-review11.json`. Независимый реальный holdout остаётся NOT_VALIDATED.
