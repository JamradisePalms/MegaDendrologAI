// user_provider.dart
import 'package:flutter/material.dart';

class UserProvider with ChangeNotifier {
  String _username = 'gringo'; // начальное значение

  String get username => _username;

  void setUsername(String username) {
    _username = username;
    notifyListeners(); // уведомляем слушателей о смене пользователя
  }
}
