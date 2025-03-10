// Copyright Take Vos 2022.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at https://www.boost.org/LICENSE_1_0.txt)

/** @file widgets/scroll_aperture_widget.hpp Defines scroll_aperture_widget.
 * @ingroup widgets
 */

#pragma once

#include "widget.hpp"

namespace hi { inline namespace v1 {

/** A scroll aperture widget.
 *
 * A widget that is used as a child of the `scroll_widget` which
 * displays a partial rectangle (the aperture) of the content.
 *
 * @ingroup widgets
 */
class scroll_aperture_widget : public widget {
public:
    using super = widget;

    observer<int> content_width;
    observer<int> content_height;
    observer<int> aperture_width;
    observer<int> aperture_height;
    observer<int> offset_x;
    observer<int> offset_y;

    scroll_aperture_widget(widget *parent) noexcept : super(parent)
    {
        hi_axiom(loop::main().on_thread());
        hi_assert_not_null(parent);

        // The aperture-widget will not draw itself, only its selected content.
        semantic_layer = parent->semantic_layer;

        _content_width_cbt = content_width.subscribe([&](auto...) {
            ++global_counter<"scroll_aperture_widget:content_width:relayout">;
            process_event({gui_event_type::window_relayout});
        });
        _content_height_cbt = content_height.subscribe([&](auto...) {
            ++global_counter<"scroll_aperture_widget:content_height:relayout">;
            process_event({gui_event_type::window_relayout});
        });
        _aperture_width_cbt = aperture_width.subscribe([&](auto...) {
            ++global_counter<"scroll_aperture_widget:aperture_width:relayout">;
            process_event({gui_event_type::window_relayout});
        });
        _aperture_height_cbt = aperture_height.subscribe([&](auto...) {
            ++global_counter<"scroll_aperture_widget:aperture_height:relayout">;
            process_event({gui_event_type::window_relayout});
        });
        _offset_x_cbt = offset_x.subscribe([&](auto...) {
            ++global_counter<"scroll_aperture_widget:offset_x:relayout">;
            process_event({gui_event_type::window_relayout});
        });
        _offset_y_cbt = offset_y.subscribe([&](auto...) {
            ++global_counter<"scroll_aperture_widget:offset_y:relayout">;
            process_event({gui_event_type::window_relayout});
        });
        _minimum_cbt = minimum.subscribe([&](auto...) {
            ++global_counter<"scroll_aperture_widget:minimum:reconstrain">;
            process_event({gui_event_type::window_reconstrain});
        });
    }

    template<typename Widget, typename... Args>
    Widget& make_widget(Args&&...args) noexcept
    {
        hi_axiom(loop::main().on_thread());
        hi_axiom(_content == nullptr);

        auto tmp = std::make_unique<Widget>(this, std::forward<Args>(args)...);
        auto& ref = *tmp;
        _content = std::move(tmp);
        return ref;
    }

    [[nodiscard]] bool x_axis_scrolls() const noexcept
    {
        return *content_width > *aperture_width;
    }

    [[nodiscard]] bool y_axis_scrolls() const noexcept
    {
        return *content_height > *aperture_height;
    }

    /// @privatesection
    [[nodiscard]] generator<widget const &> children(bool include_invisible) const noexcept override
    {
        co_yield *_content;
    }

    [[nodiscard]] box_constraints update_constraints() noexcept override
    {
        _layout = {};
        _content_constraints = _content->update_constraints();

        // The aperture can scroll so its minimum width and height are zero.
        auto aperture_constraints = _content_constraints;
        aperture_constraints.minimum = extent2i{0, 0};

        return aperture_constraints.internalize_margins().constrain(*minimum, *maximum);
    }

    void set_layout(widget_layout const& context) noexcept override
    {
        if (compare_store(_layout, context)) {
            aperture_width = context.width() - _content_constraints.margins.left() - _content_constraints.margins.right();
            aperture_height = context.height() - _content_constraints.margins.bottom() - _content_constraints.margins.top();

            // Start scrolling with the preferred size as minimum, so
            // that widgets in the content don't get unnecessarily squeezed.
            content_width = *aperture_width < _content_constraints.preferred.width() ? _content_constraints.preferred.width() :
                                                                                        *aperture_width;
            content_height = *aperture_height < _content_constraints.preferred.height() ?
                _content_constraints.preferred.height() :
                *aperture_height;
        }

        // Make sure the offsets are limited to the scrollable area.
        hilet offset_x_max = std::max(*content_width - *aperture_width, 0);
        hilet offset_y_max = std::max(*content_height - *aperture_height, 0);
        offset_x = std::clamp(*offset_x, 0, offset_x_max);
        offset_y = std::clamp(*offset_y, 0, offset_y_max);

        // The position of the content rectangle relative to the scroll view.
        // The size is further adjusted if the either the horizontal or vertical scroll bar is invisible.
        _content_shape = box_shape{
            _content_constraints,
            aarectanglei{
                -*offset_x + _content_constraints.margins.left(),
                -*offset_y + _content_constraints.margins.bottom(),
                *content_width,
                *content_height},
            theme().baseline_adjustment()};

        // The content needs to be at a higher elevation, so that hitbox check
        // will work correctly for handling scrolling with mouse wheel.
        _content->set_layout(context.transform(_content_shape, 1.0f, context.rectangle()));
    }

    void draw(draw_context const& context) noexcept override
    {
        if (*mode > widget_mode::invisible) {
            _content->draw(context);
        }
    }

    [[nodiscard]] hitbox hitbox_test(point2i position) const noexcept override
    {
        hi_axiom(loop::main().on_thread());

        if (*mode >= widget_mode::partial) {
            auto r = _content->hitbox_test_from_parent(position);

            if (layout().contains(position)) {
                r = std::max(r, hitbox{id, _layout.elevation});
            }
            return r;

        } else {
            return {};
        }
    }

    bool handle_event(gui_event const& event) noexcept override
    {
        hi_axiom(loop::main().on_thread());

        if (event == gui_event_type::mouse_wheel) {
            hilet new_offset_x = *offset_x + narrow_cast<int>(event.mouse().wheel_delta.x() * theme().scale);
            hilet new_offset_y = *offset_y + narrow_cast<int>(event.mouse().wheel_delta.y() * theme().scale);
            hilet max_offset_x = std::max(0, *content_width - *aperture_width);
            hilet max_offset_y = std::max(0, *content_height - *aperture_height);

            offset_x = std::clamp(new_offset_x, 0, max_offset_x);
            offset_y = std::clamp(new_offset_y, 0, max_offset_y);
            ++global_counter<"scroll_aperture_widget:mouse_wheel:relayout">;
            process_event({gui_event_type::window_relayout});
            return true;
        } else {
            return super::handle_event(event);
        }
    }

    void scroll_to_show(hi::aarectanglei to_show) noexcept override
    {
        if (_layout) {
            auto safe_rectangle = intersect(_layout.rectangle(), _layout.clipping_rectangle);
            int delta_x = 0;
            int delta_y = 0;

            if (safe_rectangle.width() > theme().margin<int>() * 2 and safe_rectangle.height() > theme().margin<int>() * 2) {
                // This will look visually better, if the selected widget is moved with some margin from
                // the edge of the scroll widget. The margins of the content do not have anything to do
                // with the margins that are needed here.
                safe_rectangle = safe_rectangle - theme().margin<int>();

                if (to_show.right() > safe_rectangle.right()) {
                    delta_x = to_show.right() - safe_rectangle.right();
                } else if (to_show.left() < safe_rectangle.left()) {
                    delta_x = to_show.left() - safe_rectangle.left();
                }

                if (to_show.top() > safe_rectangle.top()) {
                    delta_y = to_show.top() - safe_rectangle.top();
                } else if (to_show.bottom() < safe_rectangle.bottom()) {
                    delta_y = to_show.bottom() - safe_rectangle.bottom();
                }

                // Scroll the widget
                offset_x += delta_x;
                offset_y += delta_y;
            }

            // There may be recursive scroll view, and they all need to move until the rectangle is visible.
            if (parent) {
                parent->scroll_to_show(_layout.to_parent * translate2i(delta_x, delta_y) * to_show);
            }

        } else {
            return super::scroll_to_show(to_show);
        }
    }
    /// @endprivatesection
private:
    box_constraints _content_constraints;
    box_shape _content_shape;
    std::unique_ptr<widget> _content;
    decltype(content_width)::callback_token _content_width_cbt;
    decltype(content_height)::callback_token _content_height_cbt;
    decltype(aperture_width)::callback_token _aperture_width_cbt;
    decltype(aperture_height)::callback_token _aperture_height_cbt;
    decltype(offset_x)::callback_token _offset_x_cbt;
    decltype(offset_y)::callback_token _offset_y_cbt;
    decltype(minimum)::callback_token _minimum_cbt;
};

}} // namespace hi::v1
