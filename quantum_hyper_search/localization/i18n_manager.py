"""
Internationalization (i18n) Manager

Provides comprehensive internationalization support including message translation,
locale-specific formatting, and cultural adaptations for quantum optimization.
"""

import json
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
import re

# Try to import babel for advanced localization
try:
    from babel import Locale
    from babel.dates import format_datetime
    from babel.numbers import format_decimal, format_currency
    HAS_BABEL = True
except ImportError:
    HAS_BABEL = False

logger = logging.getLogger(__name__)


class LocalizationError(Exception):
    """Exception raised for localization errors."""
    pass


@dataclass
class LocaleConfig:
    """Configuration for a specific locale."""
    code: str                    # e.g., 'en_US', 'zh_CN'
    language: str               # e.g., 'en', 'zh'
    country: str                # e.g., 'US', 'CN'
    name: str                   # Display name
    direction: str = 'ltr'      # 'ltr' or 'rtl'
    decimal_separator: str = '.' 
    thousand_separator: str = ','
    currency_code: str = 'USD'
    date_format: str = '%Y-%m-%d'
    time_format: str = '%H:%M:%S'
    number_format: str = '#,##0.###'
    
    @classmethod
    def from_code(cls, locale_code: str) -> 'LocaleConfig':
        """Create locale config from code."""
        parts = locale_code.split('_')
        language = parts[0]
        country = parts[1] if len(parts) > 1 else language.upper()
        
        # Default configurations for common locales
        defaults = {
            'en_US': {
                'name': 'English (United States)',
                'currency_code': 'USD',
                'date_format': '%m/%d/%Y'
            },
            'en_GB': {
                'name': 'English (United Kingdom)', 
                'currency_code': 'GBP',
                'date_format': '%d/%m/%Y'
            },
            'zh_CN': {
                'name': '中文 (中国)',
                'currency_code': 'CNY',
                'date_format': '%Y年%m月%d日'
            },
            'ja_JP': {
                'name': '日本語 (日本)',
                'currency_code': 'JPY',
                'date_format': '%Y年%m月%d日'
            },
            'de_DE': {
                'name': 'Deutsch (Deutschland)',
                'currency_code': 'EUR',
                'decimal_separator': ',',
                'thousand_separator': '.',
                'date_format': '%d.%m.%Y'
            },
            'fr_FR': {
                'name': 'Français (France)',
                'currency_code': 'EUR',
                'decimal_separator': ',',
                'thousand_separator': ' ',
                'date_format': '%d/%m/%Y'
            },
            'es_ES': {
                'name': 'Español (España)',
                'currency_code': 'EUR',
                'decimal_separator': ',',
                'thousand_separator': '.',
                'date_format': '%d/%m/%Y'
            },
            'ar_SA': {
                'name': 'العربية (المملكة العربية السعودية)',
                'direction': 'rtl',
                'currency_code': 'SAR',
                'date_format': '%d/%m/%Y'
            },
            'hi_IN': {
                'name': 'हिन्दी (भारत)',
                'currency_code': 'INR',
                'date_format': '%d/%m/%Y'
            },
            'pt_BR': {
                'name': 'Português (Brasil)',
                'currency_code': 'BRL',
                'decimal_separator': ',',
                'thousand_separator': '.',
                'date_format': '%d/%m/%Y'
            }
        }
        
        config_data = defaults.get(locale_code, {})
        
        return cls(
            code=locale_code,
            language=language,
            country=country,
            name=config_data.get('name', f'{language} ({country})'),
            direction=config_data.get('direction', 'ltr'),
            decimal_separator=config_data.get('decimal_separator', '.'),
            thousand_separator=config_data.get('thousand_separator', ','),
            currency_code=config_data.get('currency_code', 'USD'),
            date_format=config_data.get('date_format', '%Y-%m-%d'),
            time_format=config_data.get('time_format', '%H:%M:%S'),
            number_format=config_data.get('number_format', '#,##0.###')
        )


class MessageCatalog:
    """Message catalog for storing translations."""
    
    def __init__(self, locale: str):
        self.locale = locale
        self.messages: Dict[str, str] = {}
        self.plural_rules: Dict[str, Dict[str, str]] = {}
        self.context_messages: Dict[str, Dict[str, str]] = {}
        
    def add_message(self, key: str, message: str, context: Optional[str] = None):
        """Add a message to the catalog."""
        if context:
            if context not in self.context_messages:
                self.context_messages[context] = {}
            self.context_messages[context][key] = message
        else:
            self.messages[key] = message
    
    def get_message(self, key: str, context: Optional[str] = None, 
                   default: Optional[str] = None) -> str:
        """Get a message from the catalog."""
        if context and context in self.context_messages:
            return self.context_messages[context].get(key, default or key)
        
        return self.messages.get(key, default or key)
    
    def add_plural(self, key: str, forms: Dict[str, str]):
        """Add plural forms for a message."""
        self.plural_rules[key] = forms
    
    def get_plural(self, key: str, count: int) -> str:
        """Get appropriate plural form."""
        if key not in self.plural_rules:
            return self.get_message(key)
        
        forms = self.plural_rules[key]
        
        # Simple English plural rule (can be extended)
        if count == 1:
            return forms.get('one', forms.get('other', key))
        else:
            return forms.get('other', forms.get('many', key))


class I18nManager:
    """
    Comprehensive internationalization manager for quantum hyperparameter optimization.
    
    Provides message translation, locale-specific formatting, and cultural adaptations
    for global deployment of quantum optimization systems.
    """
    
    def __init__(self, default_locale: str = 'en_US'):
        """Initialize i18n manager."""
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.locale_configs: Dict[str, LocaleConfig] = {}
        self.message_catalogs: Dict[str, MessageCatalog] = {}
        self.fallback_locale = 'en_US'
        
        # Initialize default locales
        self._initialize_default_locales()
        self._load_default_messages()
        
        logger.info(f"I18n manager initialized with default locale: {default_locale}")
    
    def _initialize_default_locales(self):
        """Initialize supported locales."""
        supported_locales = [
            'en_US', 'en_GB', 'zh_CN', 'ja_JP', 'de_DE', 'fr_FR', 
            'es_ES', 'ar_SA', 'hi_IN', 'pt_BR'
        ]
        
        for locale_code in supported_locales:
            self.locale_configs[locale_code] = LocaleConfig.from_code(locale_code)
            self.message_catalogs[locale_code] = MessageCatalog(locale_code)
    
    def _load_default_messages(self):
        """Load default messages for quantum optimization."""
        # Core quantum optimization messages
        messages = {
            'en_US': {
                'quantum.optimization.started': 'Quantum optimization started',
                'quantum.optimization.completed': 'Quantum optimization completed',
                'quantum.optimization.failed': 'Quantum optimization failed',
                'quantum.backend.connecting': 'Connecting to quantum backend',
                'quantum.backend.connected': 'Connected to quantum backend',
                'quantum.backend.error': 'Quantum backend error',
                'quantum.evaluation.progress': 'Evaluation progress: {progress}%',
                'quantum.result.best_score': 'Best score achieved: {score}',
                'quantum.result.best_params': 'Best parameters found',
                'quantum.error.invalid_params': 'Invalid parameters provided',
                'quantum.error.insufficient_data': 'Insufficient data for optimization',
                'quantum.warning.performance': 'Performance warning: {warning}',
                'quantum.info.iteration': 'Iteration {current}/{total}',
                'quantum.security.authenticated': 'User authenticated successfully',
                'quantum.security.access_denied': 'Access denied',
                'quantum.compliance.data_processed': 'Data processed in compliance with {regulation}',
                'ui.button.start': 'Start Optimization',
                'ui.button.stop': 'Stop Optimization',
                'ui.button.export': 'Export Results',
                'ui.label.parameters': 'Parameters',
                'ui.label.results': 'Results',
                'ui.label.progress': 'Progress',
                'status.running': 'Running',
                'status.completed': 'Completed',
                'status.failed': 'Failed',
                'time.minutes': '{count} minute(s)',
                'time.seconds': '{count} second(s)'
            },
            'zh_CN': {
                'quantum.optimization.started': '量子优化已开始',
                'quantum.optimization.completed': '量子优化已完成',
                'quantum.optimization.failed': '量子优化失败',
                'quantum.backend.connecting': '正在连接量子后端',
                'quantum.backend.connected': '已连接到量子后端',
                'quantum.backend.error': '量子后端错误',
                'quantum.evaluation.progress': '评估进度：{progress}%',
                'quantum.result.best_score': '最佳得分：{score}',
                'quantum.result.best_params': '发现最佳参数',
                'quantum.error.invalid_params': '提供的参数无效',
                'quantum.error.insufficient_data': '优化数据不足',
                'quantum.warning.performance': '性能警告：{warning}',
                'quantum.info.iteration': '迭代 {current}/{total}',
                'quantum.security.authenticated': '用户认证成功',
                'quantum.security.access_denied': '访问被拒绝',
                'quantum.compliance.data_processed': '数据处理符合{regulation}法规',
                'ui.button.start': '开始优化',
                'ui.button.stop': '停止优化',
                'ui.button.export': '导出结果',
                'ui.label.parameters': '参数',
                'ui.label.results': '结果',
                'ui.label.progress': '进度',
                'status.running': '运行中',
                'status.completed': '已完成',
                'status.failed': '失败',
                'time.minutes': '{count}分钟',
                'time.seconds': '{count}秒'
            },
            'ja_JP': {
                'quantum.optimization.started': '量子最適化が開始されました',
                'quantum.optimization.completed': '量子最適化が完了しました',
                'quantum.optimization.failed': '量子最適化が失敗しました',
                'quantum.backend.connecting': '量子バックエンドに接続中',
                'quantum.backend.connected': '量子バックエンドに接続しました',
                'quantum.backend.error': '量子バックエンドエラー',
                'quantum.evaluation.progress': '評価進行状況：{progress}%',
                'quantum.result.best_score': '最高スコア：{score}',
                'quantum.result.best_params': '最適パラメータが見つかりました',
                'quantum.error.invalid_params': '無効なパラメータが提供されました',
                'quantum.error.insufficient_data': '最適化のためのデータが不十分です',
                'quantum.warning.performance': 'パフォーマンス警告：{warning}',
                'quantum.info.iteration': 'イテレーション {current}/{total}',
                'quantum.security.authenticated': 'ユーザー認証に成功しました',
                'quantum.security.access_denied': 'アクセスが拒否されました',
                'quantum.compliance.data_processed': 'データは{regulation}規制に準拠して処理されました',
                'ui.button.start': '最適化開始',
                'ui.button.stop': '最適化停止',
                'ui.button.export': '結果エクスポート',
                'ui.label.parameters': 'パラメータ',
                'ui.label.results': '結果',
                'ui.label.progress': '進行状況',
                'status.running': '実行中',
                'status.completed': '完了',
                'status.failed': '失敗',
                'time.minutes': '{count}分',
                'time.seconds': '{count}秒'
            },
            'de_DE': {
                'quantum.optimization.started': 'Quantenoptimierung gestartet',
                'quantum.optimization.completed': 'Quantenoptimierung abgeschlossen',
                'quantum.optimization.failed': 'Quantenoptimierung fehlgeschlagen',
                'quantum.backend.connecting': 'Verbindung zum Quanten-Backend',
                'quantum.backend.connected': 'Mit Quanten-Backend verbunden',
                'quantum.backend.error': 'Quanten-Backend-Fehler',
                'quantum.evaluation.progress': 'Bewertungsfortschritt: {progress}%',
                'quantum.result.best_score': 'Beste erreichte Punktzahl: {score}',
                'quantum.result.best_params': 'Beste Parameter gefunden',
                'quantum.error.invalid_params': 'Ungültige Parameter bereitgestellt',
                'quantum.error.insufficient_data': 'Unzureichende Daten für Optimierung',
                'quantum.warning.performance': 'Leistungswarnung: {warning}',
                'quantum.info.iteration': 'Iteration {current}/{total}',
                'quantum.security.authenticated': 'Benutzer erfolgreich authentifiziert',
                'quantum.security.access_denied': 'Zugriff verweigert',
                'quantum.compliance.data_processed': 'Daten gemäß {regulation} verarbeitet',
                'ui.button.start': 'Optimierung starten',
                'ui.button.stop': 'Optimierung stoppen',
                'ui.button.export': 'Ergebnisse exportieren',
                'ui.label.parameters': 'Parameter',
                'ui.label.results': 'Ergebnisse',
                'ui.label.progress': 'Fortschritt',
                'status.running': 'Läuft',
                'status.completed': 'Abgeschlossen',
                'status.failed': 'Fehlgeschlagen',
                'time.minutes': '{count} Minute(n)',
                'time.seconds': '{count} Sekunde(n)'
            },
            'fr_FR': {
                'quantum.optimization.started': 'Optimisation quantique démarrée',
                'quantum.optimization.completed': 'Optimisation quantique terminée',
                'quantum.optimization.failed': 'Échec de l\'optimisation quantique',
                'quantum.backend.connecting': 'Connexion au backend quantique',
                'quantum.backend.connected': 'Connecté au backend quantique',
                'quantum.backend.error': 'Erreur du backend quantique',
                'quantum.evaluation.progress': 'Progrès de l\'évaluation : {progress}%',
                'quantum.result.best_score': 'Meilleur score atteint : {score}',
                'quantum.result.best_params': 'Meilleurs paramètres trouvés',
                'quantum.error.invalid_params': 'Paramètres invalides fournis',
                'quantum.error.insufficient_data': 'Données insuffisantes pour l\'optimisation',
                'quantum.warning.performance': 'Avertissement de performance : {warning}',
                'quantum.info.iteration': 'Itération {current}/{total}',
                'quantum.security.authenticated': 'Utilisateur authentifié avec succès',
                'quantum.security.access_denied': 'Accès refusé',
                'quantum.compliance.data_processed': 'Données traitées en conformité avec {regulation}',
                'ui.button.start': 'Démarrer l\'optimisation',
                'ui.button.stop': 'Arrêter l\'optimisation',
                'ui.button.export': 'Exporter les résultats',
                'ui.label.parameters': 'Paramètres',
                'ui.label.results': 'Résultats',
                'ui.label.progress': 'Progrès',
                'status.running': 'En cours',
                'status.completed': 'Terminé',
                'status.failed': 'Échoué',
                'time.minutes': '{count} minute(s)',
                'time.seconds': '{count} seconde(s)'
            },
            'es_ES': {
                'quantum.optimization.started': 'Optimización cuántica iniciada',
                'quantum.optimization.completed': 'Optimización cuántica completada',
                'quantum.optimization.failed': 'Optimización cuántica falló',
                'quantum.backend.connecting': 'Conectando al backend cuántico',
                'quantum.backend.connected': 'Conectado al backend cuántico',
                'quantum.backend.error': 'Error del backend cuántico',
                'quantum.evaluation.progress': 'Progreso de evaluación: {progress}%',
                'quantum.result.best_score': 'Mejor puntuación lograda: {score}',
                'quantum.result.best_params': 'Mejores parámetros encontrados',
                'quantum.error.invalid_params': 'Parámetros inválidos proporcionados',
                'quantum.error.insufficient_data': 'Datos insuficientes para optimización',
                'quantum.warning.performance': 'Advertencia de rendimiento: {warning}',
                'quantum.info.iteration': 'Iteración {current}/{total}',
                'quantum.security.authenticated': 'Usuario autenticado exitosamente',
                'quantum.security.access_denied': 'Acceso denegado',
                'quantum.compliance.data_processed': 'Datos procesados en cumplimiento con {regulation}',
                'ui.button.start': 'Iniciar optimización',
                'ui.button.stop': 'Detener optimización',
                'ui.button.export': 'Exportar resultados',
                'ui.label.parameters': 'Parámetros',
                'ui.label.results': 'Resultados',
                'ui.label.progress': 'Progreso',
                'status.running': 'Ejecutando',
                'status.completed': 'Completado',
                'status.failed': 'Falló',
                'time.minutes': '{count} minuto(s)',
                'time.seconds': '{count} segundo(s)'
            }
        }
        
        # Load messages into catalogs
        for locale, locale_messages in messages.items():
            if locale in self.message_catalogs:
                catalog = self.message_catalogs[locale]
                for key, message in locale_messages.items():
                    catalog.add_message(key, message)
    
    def set_locale(self, locale_code: str) -> bool:
        """Set the current locale."""
        if locale_code in self.locale_configs:
            self.current_locale = locale_code
            logger.info(f"Locale set to: {locale_code}")
            return True
        else:
            logger.warning(f"Unsupported locale: {locale_code}")
            return False
    
    def get_current_locale(self) -> str:
        """Get the current locale code."""
        return self.current_locale
    
    def get_locale_config(self, locale_code: Optional[str] = None) -> LocaleConfig:
        """Get locale configuration."""
        locale = locale_code or self.current_locale
        return self.locale_configs.get(locale, self.locale_configs[self.default_locale])
    
    def get_supported_locales(self) -> List[str]:
        """Get list of supported locales."""
        return list(self.locale_configs.keys())
    
    def translate(self, key: str, locale: Optional[str] = None, 
                 context: Optional[str] = None, **kwargs) -> str:
        """Translate a message key."""
        target_locale = locale or self.current_locale
        
        # Try target locale first
        if target_locale in self.message_catalogs:
            message = self.message_catalogs[target_locale].get_message(key, context)
            if message != key:  # Found translation
                return self._format_message(message, **kwargs)
        
        # Try fallback locale
        if self.fallback_locale in self.message_catalogs and target_locale != self.fallback_locale:
            message = self.message_catalogs[self.fallback_locale].get_message(key, context)
            if message != key:  # Found fallback translation
                return self._format_message(message, **kwargs)
        
        # Return key as last resort
        return self._format_message(key, **kwargs)
    
    def translate_plural(self, key: str, count: int, locale: Optional[str] = None, **kwargs) -> str:
        """Translate a message with plural forms."""
        target_locale = locale or self.current_locale
        
        if target_locale in self.message_catalogs:
            message = self.message_catalogs[target_locale].get_plural(key, count)
            return self._format_message(message, count=count, **kwargs)
        
        # Fallback
        return self._format_message(key, count=count, **kwargs)
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with parameters."""
        try:
            return message.format(**kwargs)
        except (KeyError, ValueError) as e:
            logger.warning(f"Message formatting error: {e}")
            return message
    
    def format_number(self, value: float, locale: Optional[str] = None) -> str:
        """Format number according to locale conventions."""
        config = self.get_locale_config(locale)
        
        if HAS_BABEL:
            try:
                locale_obj = Locale.parse(config.code.replace('_', '-'))
                return format_decimal(value, locale=locale_obj)
            except Exception:
                pass
        
        # Fallback formatting
        formatted = f"{value:,.3f}"
        if config.decimal_separator != '.':
            formatted = formatted.replace('.', 'TEMP').replace(',', config.thousand_separator).replace('TEMP', config.decimal_separator)
        
        return formatted
    
    def format_currency(self, amount: float, currency: Optional[str] = None, 
                       locale: Optional[str] = None) -> str:
        """Format currency according to locale conventions."""
        config = self.get_locale_config(locale)
        currency_code = currency or config.currency_code
        
        if HAS_BABEL:
            try:
                locale_obj = Locale.parse(config.code.replace('_', '-'))
                return format_currency(amount, currency_code, locale=locale_obj)
            except Exception:
                pass
        
        # Fallback formatting
        formatted_amount = self.format_number(amount, locale)
        return f"{currency_code} {formatted_amount}"
    
    def format_datetime(self, dt: datetime, locale: Optional[str] = None, 
                       format_type: str = 'medium') -> str:
        """Format datetime according to locale conventions."""
        config = self.get_locale_config(locale)
        
        if HAS_BABEL:
            try:
                locale_obj = Locale.parse(config.code.replace('_', '-'))
                return format_datetime(dt, format=format_type, locale=locale_obj)
            except Exception:
                pass
        
        # Fallback formatting
        if format_type == 'date':
            return dt.strftime(config.date_format)
        elif format_type == 'time':
            return dt.strftime(config.time_format)
        else:
            return dt.strftime(f"{config.date_format} {config.time_format}")
    
    def get_text_direction(self, locale: Optional[str] = None) -> str:
        """Get text direction for locale."""
        config = self.get_locale_config(locale)
        return config.direction
    
    def is_rtl(self, locale: Optional[str] = None) -> bool:
        """Check if locale uses right-to-left text direction."""
        return self.get_text_direction(locale) == 'rtl'
    
    def add_translation(self, locale: str, key: str, message: str, context: Optional[str] = None):
        """Add a translation dynamically."""
        if locale not in self.message_catalogs:
            self.message_catalogs[locale] = MessageCatalog(locale)
        
        self.message_catalogs[locale].add_message(key, message, context)
        logger.debug(f"Added translation for {locale}: {key} = {message}")
    
    def load_translations_from_file(self, locale: str, file_path: Union[str, Path]):
        """Load translations from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                translations = json.load(f)
            
            if locale not in self.message_catalogs:
                self.message_catalogs[locale] = MessageCatalog(locale)
            
            catalog = self.message_catalogs[locale]
            
            for key, value in translations.items():
                if isinstance(value, dict):
                    # Handle context or plural forms
                    if 'context' in value:
                        catalog.add_message(key, value['message'], value['context'])
                    elif 'one' in value or 'other' in value:
                        catalog.add_plural(key, value)
                else:
                    catalog.add_message(key, value)
            
            logger.info(f"Loaded {len(translations)} translations for {locale} from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load translations from {file_path}: {e}")
            raise LocalizationError(f"Failed to load translations: {e}")
    
    def export_translations(self, locale: str, file_path: Union[str, Path]):
        """Export translations to JSON file."""
        try:
            if locale not in self.message_catalogs:
                raise LocalizationError(f"No translations available for locale: {locale}")
            
            catalog = self.message_catalogs[locale]
            translations = catalog.messages.copy()
            
            # Add context messages
            for context, context_messages in catalog.context_messages.items():
                for key, message in context_messages.items():
                    translations[f"{context}.{key}"] = message
            
            # Add plural forms
            for key, plural_forms in catalog.plural_rules.items():
                translations[f"{key}_plural"] = plural_forms
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Exported {len(translations)} translations for {locale} to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export translations to {file_path}: {e}")
            raise LocalizationError(f"Failed to export translations: {e}")
    
    def get_translation_coverage(self, locale: str) -> float:
        """Get translation coverage percentage for a locale."""
        if locale not in self.message_catalogs:
            return 0.0
        
        if self.fallback_locale not in self.message_catalogs:
            return 1.0  # Can't calculate coverage without baseline
        
        fallback_catalog = self.message_catalogs[self.fallback_locale]
        target_catalog = self.message_catalogs[locale]
        
        total_keys = len(fallback_catalog.messages)
        if total_keys == 0:
            return 1.0
        
        translated_keys = sum(1 for key in fallback_catalog.messages.keys() 
                            if target_catalog.get_message(key) != key)
        
        return translated_keys / total_keys
    
    def validate_translations(self, locale: str) -> List[str]:
        """Validate translations and return list of issues."""
        issues = []
        
        if locale not in self.message_catalogs:
            issues.append(f"No message catalog for locale: {locale}")
            return issues
        
        catalog = self.message_catalogs[locale]
        
        # Check for empty messages
        for key, message in catalog.messages.items():
            if not message.strip():
                issues.append(f"Empty message for key: {key}")
        
        # Check for untranslated keys (same as key)
        if self.fallback_locale in self.message_catalogs and locale != self.fallback_locale:
            fallback_catalog = self.message_catalogs[self.fallback_locale]
            for key in fallback_catalog.messages.keys():
                if catalog.get_message(key) == key:
                    issues.append(f"Untranslated key: {key}")
        
        # Check for format string consistency
        for key, message in catalog.messages.items():
            placeholders = re.findall(r'\{([^}]+)\}', message)
            if self.fallback_locale in self.message_catalogs:
                fallback_message = self.message_catalogs[self.fallback_locale].get_message(key)
                fallback_placeholders = re.findall(r'\{([^}]+)\}', fallback_message)
                
                if set(placeholders) != set(fallback_placeholders):
                    issues.append(f"Format string mismatch for key {key}: {placeholders} vs {fallback_placeholders}")
        
        return issues
    
    def get_localization_summary(self) -> Dict[str, Any]:
        """Get comprehensive localization summary."""
        summary = {
            'current_locale': self.current_locale,
            'default_locale': self.default_locale,
            'fallback_locale': self.fallback_locale,
            'supported_locales': self.get_supported_locales(),
            'locale_configs': {}
        }
        
        # Add locale-specific information
        for locale_code, config in self.locale_configs.items():
            locale_summary = {
                'name': config.name,
                'direction': config.direction,
                'currency': config.currency_code,
                'date_format': config.date_format,
                'messages_count': len(self.message_catalogs[locale_code].messages) if locale_code in self.message_catalogs else 0,
                'translation_coverage': self.get_translation_coverage(locale_code),
                'validation_issues': len(self.validate_translations(locale_code))
            }
            summary['locale_configs'][locale_code] = locale_summary
        
        return summary
    
    # Convenience methods for common quantum optimization messages
    def msg_optimization_started(self, **kwargs) -> str:
        """Get localized 'optimization started' message."""
        return self.translate('quantum.optimization.started', **kwargs)
    
    def msg_optimization_completed(self, **kwargs) -> str:
        """Get localized 'optimization completed' message."""
        return self.translate('quantum.optimization.completed', **kwargs)
    
    def msg_optimization_failed(self, **kwargs) -> str:
        """Get localized 'optimization failed' message."""
        return self.translate('quantum.optimization.failed', **kwargs)
    
    def msg_progress(self, progress: float, **kwargs) -> str:
        """Get localized progress message."""
        return self.translate('quantum.evaluation.progress', progress=progress, **kwargs)
    
    def msg_best_score(self, score: float, **kwargs) -> str:
        """Get localized best score message."""
        formatted_score = self.format_number(score)
        return self.translate('quantum.result.best_score', score=formatted_score, **kwargs)
    
    def msg_iteration(self, current: int, total: int, **kwargs) -> str:
        """Get localized iteration message."""
        return self.translate('quantum.info.iteration', current=current, total=total, **kwargs)
