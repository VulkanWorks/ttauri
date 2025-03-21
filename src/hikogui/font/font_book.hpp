// Copyright Take Vos 2020-2022.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "font.hpp"
#include "font_family_id.hpp"
#include "font_grapheme_id.hpp"
#include "glyph_ids.hpp"
#include "elusive_icon.hpp"
#include "hikogui_icon.hpp"
#include "../unicode/grapheme.hpp"
#include "../geometry/module.hpp"
#include "../utility/module.hpp"
#include "../generator.hpp"
#include <limits>
#include <array>
#include <new>
#include <atomic>
#include <filesystem>

namespace hi::inline v1 {

/** font_book keeps track of multiple fonts.
 * The font_book is instantiated during application startup
 * and is available through Foundation_globals->font_book.
 *
 *
 */
class font_book {
public:
    static font_book& global() noexcept;

    ~font_book();
    font_book(font_book const&) = delete;
    font_book(font_book&&) = delete;
    font_book& operator=(font_book const&) = delete;
    font_book& operator=(font_book&&) = delete;
    font_book();

    /** Register a font.
     * Duplicate registrations will be ignored.
     *
     * When a font file is registered the file will be temporarily opened to read and cache a set of properties:
     *  - The English font Family from the 'name' table.
     *  - The weight, width, slant & design-size from the 'fdsc' table.
     *  - The character map 'cmap' table.
     *
     * @param path Location of font.
     * @param post_process Calculate font fallback
     */
    font& register_font(std::filesystem::path const& path, bool post_process = true);

    /** Register all fonts found in a directory.
     *
     * @see register_font()
     */
    void register_font_directory(std::filesystem::path const& path, bool post_process = true);

    void register_elusive_icon_font(std::filesystem::path const& path)
    {
        _elusive_icon_font = &register_font(path, false);
    }

    void register_hikogui_icon_font(std::filesystem::path const& path)
    {
        _hikogui_icon_font = &register_font(path, false);
    }

    /** Post process font_book
     * Should be called after a set of register_font() calls
     * This calculates font fallbacks.
     */
    void post_process() noexcept;

    /** Find font family id.
     * This function will always return a valid font_family_id by walking the fallback-chain.
     */
    [[nodiscard]] font_family_id find_family(std::string_view family_name) const noexcept;

    /** Register font family id.
     * If the family already exists the existing family_id is returned.
     */
    [[nodiscard]] font_family_id register_family(std::string_view family_name) noexcept;

    /** Find a font closest to the variant.
     * This function will always return a valid font_id.
     *
     * @param family_id a valid family id.
     * @param variant The variant of the font to select.
     * @return a valid font id.
     */
    [[nodiscard]] font const& find_font(font_family_id family_id, font_variant variant) const noexcept;

    /** Find a font closest to the variant.
     * This function will always return a valid font_id.
     *
     * @param family_id a valid family id.
     * @param weight The weight of the font to select.
     * @param italic If the font to select should be italic or not.
     * @return a valid font id.
     */
    [[nodiscard]] font const& find_font(font_family_id family_id, font_weight weight, bool italic) const noexcept;

    /** Find a font closest to the variant.
     * This function will always return a valid font_id.
     *
     * @param family_name A name of a font family, which may be invalid.
     * @param weight The weight of the font to select.
     * @param italic If the font to select should be italic or not.
     * @return a font id, possibly from a fallback font.
     */
    [[nodiscard]] font const& find_font(std::string_view family_name, font_weight weight, bool italic) const noexcept;

    /** Find a glyph using the given code-point.
     * This function will find a glyph matching the grapheme in the selected font, or
     * find the glyph in the fallback font.
     *
     * @param font The font to use to find the grapheme in.
     * @param grapheme The Unicode grapheme to find in the font.
     * @return A list of glyphs which matched the grapheme.
     */
    [[nodiscard]] glyph_ids find_glyph(font const& font, grapheme grapheme) const noexcept;

    [[nodiscard]] glyph_ids find_glyph(elusive_icon rhs) const noexcept
    {
        hi_assert_not_null(_elusive_icon_font);
        return {*_elusive_icon_font, _elusive_icon_font->find_glyph(grapheme{static_cast<char32_t>(rhs)})};
    }

    [[nodiscard]] glyph_ids find_glyph(hikogui_icon rhs) const noexcept
    {
        hi_assert_not_null(_hikogui_icon_font);
        return {*_hikogui_icon_font, _hikogui_icon_font->find_glyph(grapheme{static_cast<char32_t>(rhs)})};
    }

    struct estimate_run_result_type {
        /** The resolved font to use for each grapheme.
         *
         * If a grapheme is not available to be displayed by a font, then
         * a fallback font is searched. Use this particular font when
         * text-shaping a run.
         */
        std::vector<font const *> fonts;

        /** The estimated advance for each grapheme.
         *
         * This advance is used in the line folding algorithm.
         */
        std::vector<float> advances;

        void reserve(size_t count) noexcept
        {
            fonts.reserve(count);
            advances.reserve(count);
        }

        void scale(float s) noexcept
        {
            for (auto& advance : advances) {
                advance *= s;
            }
        }
    };

    /** Estimate a run of text.
     *
     * This function is used by the text shaper to estimate the advance for
     * each grapheme in a run (same style, size, color, font, language, script).
     * 
     * @param The font for this run of text.
     * @param run The run of text.
     * @return A list of resolved fonts for each grapheme, A list of estimated advance for each grapheme.
     */
    [[nodiscard]] estimate_run_result_type estimate_run(font const& font, gstring run) const noexcept;

private:
    inline static std::unique_ptr<font_book> _global = nullptr;

    font const *_elusive_icon_font = nullptr;
    font const *_hikogui_icon_font = nullptr;

    /** Table of font_family_ids index using the family-name.
     */
    std::unordered_map<std::string, font_family_id> _family_names;

    /** A list of family name -> fallback family name
     */
    std::unordered_map<std::string, std::string> _family_name_fallback_chain;

    /** Different fonts; variants of a family.
     */
    std::vector<std::array<font const *, font_variant::max()>> _font_variants;

    std::vector<std::unique_ptr<font>> _fonts;
    std::vector<hi::font *> _font_ptrs;

    /** Same as family_name, but will also have resolved font families from the fallback_chain.
     * Must be cleared when a new font family is registered.
     */
    mutable std::unordered_map<std::string, font_family_id> _family_name_cache;

    /**
     * Must be cleared when a new font is registered.
     */
    mutable std::unordered_map<font_grapheme_id, glyph_ids> _glyph_cache;

    [[nodiscard]] std::vector<hi::font *> make_fallback_chain(font_weight weight, bool italic) noexcept;

    /** Generate fallback font family names.
     */
    [[nodiscard]] generator<std::string> generate_family_names(std::string_view name) const noexcept;

    void create_family_name_fallback_chain() noexcept;
};

/** Find a glyph using the given code-point.
 * This function will find a glyph matching the grapheme in the selected font, or
 * find the glyph in the fallback font.
 *
 * @param font The font to use to find the grapheme in.
 * @param grapheme The Unicode grapheme to find in the font.
 * @return A list of glyphs which matched the grapheme.
 */
[[nodiscard]] inline glyph_ids find_glyph(font const& font, grapheme grapheme) noexcept
{
    return font_book::global().find_glyph(font, grapheme);
}

[[nodiscard]] inline glyph_ids find_glyph(elusive_icon rhs) noexcept
{
    return font_book::global().find_glyph(rhs);
}

[[nodiscard]] inline glyph_ids find_glyph(hikogui_icon rhs) noexcept
{
    return font_book::global().find_glyph(rhs);
}

} // namespace hi::inline v1
