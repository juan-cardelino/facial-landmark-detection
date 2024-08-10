extends CanvasLayer

signal cambiar_modo

var modo:String = 'nada'

func _on_minor_pressed():
	change_mode('minor')

func _on_belt_pressed():
	change_mode('belt')

func _on_adder_pressed():
	change_mode('adder')


func change_mode(control):
	if modo != control:
		modo = control
	else:
		modo = 'nada'
	emit_signal("cambiar_modo")
