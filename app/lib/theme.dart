// lib/theme.dart
import 'package:flutter/material.dart';

// Основная цветовая палитра
const Color primaryColor = Color(0xFF4CAF50);       // зелёный
const Color primaryColorDark = Color(0xFF388E3C);   // тёмно-зелёный
const Color secondaryColor = Color(0xFFFFB74D);     // оранжевый акцент
const Color backgroundColor = Color(0xFFF1F8E9);    // светлый фон
const Color surfaceColor = Color(0xFFFFFFFF);       // цвет карточек
const Color errorColor = Color(0xFFE53935);         // ошибки
const Color onPrimary = Color(0xFFFFFFFF);          // текст на кнопках
const Color onBackground = Color(0xFF212121);      // основной текст
const Color onSurface = Color(0xFF212121);         // текст на карточках

final ThemeData appTheme = ThemeData(
  brightness: Brightness.light,
  primaryColor: primaryColor,
  primaryColorDark: primaryColorDark,
  scaffoldBackgroundColor: backgroundColor,

  // Убираем эту строку:
  // errorColor: errorColor,  

  colorScheme: ColorScheme(
    primary: primaryColor,
    primaryContainer: primaryColorDark,
    secondary: secondaryColor,
    secondaryContainer: secondaryColor,
    surface: surfaceColor,
    background: backgroundColor,
    error: errorColor,            // здесь определяем цвет ошибок
    onPrimary: onPrimary,
    onSecondary: onBackground,
    onSurface: onSurface,
    onBackground: onBackground,
    onError: onPrimary,
    brightness: Brightness.light,
  ),

  elevatedButtonTheme: ElevatedButtonThemeData(
    style: ElevatedButton.styleFrom(
      backgroundColor: primaryColor,
      foregroundColor: onPrimary,
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
      textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
    ),
  ),

  appBarTheme: const AppBarTheme(
    backgroundColor: primaryColor,
    foregroundColor: onPrimary,
    elevation: 2,
    centerTitle: true,
  ),

  textTheme: const TextTheme(
    bodyMedium: TextStyle(color: onBackground, fontSize: 16),
    bodyLarge: TextStyle(color: onBackground, fontSize: 18, fontWeight: FontWeight.w600),
    titleLarge: TextStyle(color: onBackground, fontSize: 20, fontWeight: FontWeight.bold),
  ),

  snackBarTheme: SnackBarThemeData(
    backgroundColor: primaryColorDark,
    contentTextStyle: const TextStyle(color: onPrimary),
  ),
);

